"""Managed agent runs shared by quick chat, streaming chat and AIOps."""

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.agent_run_service import RunConflict, agent_run_service

router = APIRouter()


class RunRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    mode: Literal["quick", "stream", "aiops"] = "stream"
    question: str = Field(default="", max_length=100000)


def find_run(run_id: str):
    try:
        return agent_run_service.get(run_id)
    except KeyError:
        raise HTTPException(404, "任务不存在或已过期，请重新发送问题") from None


@router.post("/agent/runs", status_code=201)
async def start_run(request: RunRequest):
    if request.mode != "aiops" and not request.question.strip():
        raise HTTPException(422, "问题不能为空")

    turn_id = uuid4().hex

    def factory(resume):
        if request.mode == "aiops":
            from app.services.aiops_service import aiops_service
            return aiops_service.diagnose(turn_id, resume=resume)
        from app.services.rag_agent_service import rag_agent_service
        return rag_agent_service.query_stream(
            request.question, request.session_id, resume=resume, turn_id=turn_id
        )

    try:
        run = agent_run_service.start(request.session_id, request.mode, request.question, factory)
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return run.snapshot()


@router.get("/agent/runs/{run_id}")
async def run_status(run_id: str):
    return find_run(run_id).snapshot()


@router.post("/agent/runs/{run_id}/stop")
async def stop_run(run_id: str):
    find_run(run_id)
    return (await agent_run_service.stop(run_id)).snapshot()


@router.post("/agent/runs/{run_id}/resume")
async def resume_run(run_id: str):
    find_run(run_id)
    try:
        return agent_run_service.resume(run_id).snapshot()
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from exc
