import asyncio
import sys
import types
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from app.api import agent_runs
from app.services.agent_run_service import AgentRunService


class AgentRunApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_stop_resume_status_and_validation(self):
        service = AgentRunService()
        app = FastAPI()
        app.include_router(agent_runs.router, prefix="/api")
        entered = asyncio.Event()
        seen = []

        async def query_stream(question, session_id, resume=False, turn_id=None):
            seen.append((question, session_id, resume, turn_id))
            yield {"type": "content", "data": "resumed" if resume else "partial"}
            if not resume:
                entered.set()
                await asyncio.Event().wait()

        module = types.ModuleType("app.services.rag_agent_service")
        module.rag_agent_service = types.SimpleNamespace(query_stream=query_stream)
        with patch.object(agent_runs, "agent_run_service", service), patch.dict(sys.modules, {module.__name__: module}):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                created = await client.post("/api/agent/runs", json={"session_id": "one", "mode": "stream", "question": "time?"})
                self.assertEqual(created.status_code, 201)
                run_id = created.json()["id"]
                await asyncio.wait_for(entered.wait(), 2)
                conflict = await client.post("/api/agent/runs", json={"session_id": "one", "question": "other"})
                self.assertEqual(conflict.status_code, 409)
                stopped = await client.post(f"/api/agent/runs/{run_id}/stop")
                self.assertEqual(stopped.json()["status"], "paused")
                self.assertEqual(stopped.json()["answer"], "partial")
                resumed = await client.post(f"/api/agent/runs/{run_id}/resume")
                self.assertEqual(resumed.status_code, 200)
                await service.get(run_id).task
                status = (await client.get(f"/api/agent/runs/{run_id}")).json()
                self.assertEqual((status["status"], status["answer"]), ("completed", "resumed"))
                self.assertEqual(seen[0][3], seen[1][3])
                self.assertEqual([call[2] for call in seen], [False, True])
                self.assertEqual((await client.post(f"/api/agent/runs/{run_id}/resume")).status_code, 409)
                self.assertEqual((await client.get("/api/agent/runs/missing")).status_code, 404)
                self.assertEqual((await client.post("/api/agent/runs", json={"session_id": "x", "question": " "})).status_code, 422)
        await service.shutdown()


if __name__ == "__main__":
    unittest.main()
