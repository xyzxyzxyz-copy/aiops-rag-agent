"""Process-local task control. Cancellation is acknowledged before resuming a graph."""

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, AsyncIterator, Callable
from uuid import uuid4


class RunConflict(ValueError):
    pass


@dataclass
class AgentRun:
    id: str
    session_id: str
    mode: str
    question: str
    factory: Callable[[bool], AsyncIterator[dict]] = field(repr=False)
    status: str = "running"
    answer: str = ""
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    version: int = 0
    updated: float = field(default_factory=monotonic)
    task: asyncio.Task | None = field(default=None, repr=False)

    def touch(self):
        self.version += 1
        self.updated = monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in (
            "id", "session_id", "mode", "status", "answer", "error", "version"
        )} | {"events": list(self.events)}


class AgentRunService:
    def __init__(self, max_runs: int = 200, ttl: float = 3600):
        self.runs: dict[str, AgentRun] = {}
        self.max_runs = max_runs
        self.ttl = ttl

    def start(self, session_id: str, mode: str, question: str, factory) -> AgentRun:
        for run_id, run in list(self.runs.items()):
            if run.status in {"completed", "failed", "paused"} and monotonic() - run.updated > self.ttl:
                del self.runs[run_id]
        if any(r.session_id == session_id and r.status in {"running", "stopping", "paused"}
               for r in self.runs.values()):
            raise RunConflict("当前会话仍有运行中或已停止的任务，请继续任务或新建对话")
        if len(self.runs) >= self.max_runs:
            terminal = next((r.id for r in self.runs.values()
                             if r.status in {"completed", "failed"}), None)
            if terminal is None:
                raise RunConflict("任务容量已满，请稍后重试")
            del self.runs[terminal]
        run = AgentRun(uuid4().hex, session_id, mode, question, factory)
        self.runs[run.id] = run
        self._launch(run, False)
        return run

    def get(self, run_id: str) -> AgentRun:
        return self.runs[run_id]

    def _launch(self, run: AgentRun, resume: bool):
        run.status = "running"
        run.error = None
        # A cancelled model node may generate its answer again on resume.
        if resume and run.mode != "aiops":
            run.answer = ""
        run.touch()
        run.task = asyncio.create_task(self._execute(run, resume))
        run.task.add_done_callback(lambda task: self._settled(run, task))

    def _settled(self, run: AgentRun, task: asyncio.Task):
        # Also handles cancellation before the coroutine's first instruction.
        if task.cancelled() and run.status == "stopping":
            run.status = "paused"
            run.touch()
        elif not task.cancelled():
            task.exception()  # Retrieve unexpected failures; never leave an unhandled task.

    async def _execute(self, run: AgentRun, resume: bool):
        try:
            async with aclosing(run.factory(resume)) as events:
                async for event in events:
                    if run.status == "stopping":
                        raise asyncio.CancelledError
                    kind = event.get("type")
                    if kind == "error":
                        raise RuntimeError(event.get("data") or event.get("message") or "任务执行失败")
                    if run.mode == "aiops":
                        run.events.append(event)
                        if kind == "complete":
                            run.answer = event.get("response") or event.get("diagnosis", {}).get("report", "")
                    elif kind == "content":
                        run.answer += str(event.get("data") or "")
                    run.touch()
            run.status = "paused" if run.status == "stopping" else "completed"
        except asyncio.CancelledError:
            run.status = "paused"
            raise
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
        finally:
            run.touch()

    async def stop(self, run_id: str) -> AgentRun:
        run = self.get(run_id)
        if run.status == "running":
            run.status = "stopping"
            run.touch()
            run.task.cancel()
        if run.status == "stopping":
            # Do not claim the task has stopped until cancellation propagates.
            await asyncio.wait({run.task}, timeout=3)
            if run.task.done() and run.status == "stopping":
                self._settled(run, run.task)
        return run

    def resume(self, run_id: str) -> AgentRun:
        run = self.get(run_id)
        if run.status != "paused" or not run.task.done():
            raise RunConflict("只有已停止的任务可以继续，请等待后台确认停止")
        self._launch(run, True)
        return run

    async def shutdown(self):
        tasks = [self.stop(r.id) for r in self.runs.values() if r.status in {"running", "stopping"}]
        if tasks:
            await asyncio.gather(*tasks)


agent_run_service = AgentRunService()
