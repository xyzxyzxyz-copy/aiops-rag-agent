"""Offline cancellation tests: python -m unittest discover -s tests -v."""

import asyncio
import unittest
from unittest.mock import patch

from app.services.agent_run_service import AgentRunService, RunConflict


class AgentRunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = AgentRunService()

    async def asyncTearDown(self):
        await self.service.shutdown()

    async def test_stop_cancels_work_and_resume_replaces_partial_answer(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        calls = []

        async def factory(resume):
            calls.append(resume)
            yield {"type": "content", "data": "完整回答" if resume else "半句"}
            if not resume:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        run = self.service.start("session", "stream", "question", factory)
        await entered.wait()
        await self.service.stop(run.id)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(run.task.done())
        self.assertEqual((run.status, run.answer), ("paused", "半句"))
        self.service.resume(run.id)
        with self.assertRaises(RunConflict):
            self.service.resume(run.id)
        await run.task
        self.assertEqual((run.status, run.answer), ("completed", "完整回答"))
        self.assertEqual(calls, [False, True])

    async def test_stop_before_coroutine_started_and_repeat_stop(self):
        async def factory(resume):
            yield {"type": "content", "data": "hello"}

        run = self.service.start("session", "quick", "question", factory)
        await self.service.stop(run.id)
        await self.service.stop(run.id)
        self.assertEqual(run.status, "paused")
        self.service.resume(run.id)
        await run.task
        await self.service.stop(run.id)
        self.assertEqual(run.status, "completed")

    async def test_one_task_per_session_and_session_isolation(self):
        async def factory(resume):
            await asyncio.Event().wait()
            yield {}

        first = self.service.start("one", "quick", "q", factory)
        second = self.service.start("two", "aiops", "", factory)
        with self.assertRaises(RunConflict):
            self.service.start("one", "aiops", "", factory)
        await self.service.stop(first.id)
        self.assertEqual(second.status, "running")
        with self.assertRaises(RunConflict):
            self.service.start("one", "quick", "q", factory)

    async def test_error_does_not_look_completed(self):
        async def factory(resume):
            yield {"type": "error", "data": "tool failed"}

        run = self.service.start("session", "stream", "q", factory)
        await run.task
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.error, "tool failed")

    async def test_aiops_retains_completed_steps_on_resume(self):
        entered = asyncio.Event()

        async def factory(resume):
            if not resume:
                yield {"type": "step_complete", "message": "step one"}
                entered.set()
                await asyncio.Event().wait()
            else:
                yield {"type": "complete", "diagnosis": {"report": "report"}}

        run = self.service.start("session", "aiops", "", factory)
        await entered.wait()
        await self.service.stop(run.id)
        self.service.resume(run.id)
        await run.task
        self.assertEqual(run.answer, "report")
        self.assertEqual([e["type"] for e in run.events], ["step_complete", "complete"])

    async def test_capacity_and_expiry(self):
        service = AgentRunService(max_runs=1, ttl=10)

        async def factory(resume):
            yield {"type": "content", "data": "done"}

        old = service.start("old", "quick", "q", factory)
        with self.assertRaises(RunConflict):
            service.start("new", "quick", "q", factory)
        await old.task
        new = service.start("new", "quick", "q", factory)
        self.assertNotIn(old.id, service.runs)
        await service.stop(new.id)
        new.updated -= 11
        replacement = service.start("new", "quick", "q", factory)
        await replacement.task

    async def test_cannot_resume_until_cancellation_cleanup_finishes(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def factory(resume):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()  # Simulate a transport closing asynchronously.
            if False:
                yield {}

        run = self.service.start("session", "stream", "q", factory)
        await entered.wait()
        real_wait = asyncio.wait

        async def short_wait(tasks, timeout):
            return await real_wait(tasks, timeout=0.01)

        with patch("app.services.agent_run_service.asyncio.wait", short_wait):
            await self.service.stop(run.id)
        self.assertEqual(run.status, "stopping")
        with self.assertRaises(RunConflict):
            self.service.resume(run.id)
        release.set()
        await run.task
        self.assertEqual(run.status, "paused")


if __name__ == "__main__":
    unittest.main()
