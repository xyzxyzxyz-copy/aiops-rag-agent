"""Exercise the real LangGraph checkpointer with a local fake streaming model."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.services.agent_run_service import AgentRunService


class CheckpointResumeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Loading production RAG code should not initialize a vector database in tests.
        tools = types.ModuleType("app.tools")
        tools.get_current_time = MagicMock()
        tools.retrieve_knowledge = MagicMock()
        spec = importlib.util.spec_from_file_location(
            "rag_resume_under_test", Path(__file__).resolve().parents[1] / "app/services/rag_agent_service.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.tools": tools}), patch("langchain_qwq.ChatQwen"):
            spec.loader.exec_module(module)
        self.rag = module.RagAgentService.__new__(module.RagAgentService)
        self.rag._initialize_agent = AsyncMock()
        self.rag.system_prompt = "test"
        self.tool_calls = 0
        model = FakeListChatModel(responses=["完整答案"], sleep=0.03)

        async def tool_node(state):
            self.tool_calls += 1
            return {"messages": []}

        async def model_node(state):
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("tools", tool_node)
        graph.add_node("model", model_node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", "model")
        graph.add_edge("model", END)
        self.rag.agent = graph.compile(checkpointer=MemorySaver())
        self.service = AgentRunService()

    async def asyncTearDown(self):
        await self.service.shutdown()

    async def test_cancel_resume_skips_completed_tool_node(self):
        def factory(resume):
            return self.rag.query_stream("time?", "session", resume=resume, turn_id="turn-one")

        run = self.service.start("session", "stream", "time?", factory)

        async def wait_for_output():
            while not run.answer and run.status == "running":
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_output(), 5)
        self.assertTrue(run.answer)
        await self.service.stop(run.id)
        self.assertEqual(run.status, "paused")
        self.assertEqual(self.tool_calls, 1)
        self.service.resume(run.id)
        await asyncio.wait_for(run.task, 5)
        self.assertEqual((run.status, run.answer), ("completed", "完整答案"))
        self.assertEqual(self.tool_calls, 1)

    async def test_early_stop_new_turn_does_not_resume_previous_answer(self):
        async for _ in self.rag.query_stream("first", "session", turn_id="old-turn"):
            pass
        self.assertEqual(self.tool_calls, 1)

        def factory(resume):
            return self.rag.query_stream("second", "session", resume=resume, turn_id="new-turn")

        run = self.service.start("session", "quick", "second", factory)
        await self.service.stop(run.id)  # No new graph checkpoint exists yet.
        self.service.resume(run.id)
        await asyncio.wait_for(run.task, 5)
        self.assertEqual(run.status, "completed")
        self.assertEqual(self.tool_calls, 2)

    async def test_aiops_real_graph_resumes_remaining_steps(self):
        tools = types.ModuleType("app.tools")
        tools.get_current_time = MagicMock()
        tools.retrieve_knowledge = MagicMock()
        spec = importlib.util.spec_from_file_location(
            "aiops_resume_under_test", Path(__file__).resolve().parents[1] / "app/services/aiops_service.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.tools": tools}):
            spec.loader.exec_module(module)
        started_second = asyncio.Event()
        release_second = asyncio.Event()
        calls = []

        async def planner(state):
            calls.append("plan")
            return {"plan": ["one", "two"]}

        async def executor(state):
            step = state["plan"][0]
            calls.append(step)
            if step == "two":
                started_second.set()
                await release_second.wait()
            return {"plan": state["plan"][1:], "past_steps": [(step, "ok")]}

        async def replanner(state):
            return {} if state["plan"] else {"response": "final report"}

        module.planner, module.executor, module.replanner = planner, executor, replanner
        aiops = module.AIOpsService()

        def factory(resume):
            return aiops.diagnose("aiops-thread", resume=resume)

        run = self.service.start("aiops-session", "aiops", "", factory)
        await asyncio.wait_for(started_second.wait(), 5)
        await self.service.stop(run.id)
        self.assertEqual(run.status, "paused")
        release_second.set()
        self.service.resume(run.id)
        await asyncio.wait_for(run.task, 5)
        self.assertEqual((run.status, run.answer), ("completed", "final report"))
        self.assertEqual(calls.count("plan"), 1)
        self.assertEqual(calls.count("one"), 1)
        self.assertEqual(calls.count("two"), 2)  # Only the interrupted step is retried.


if __name__ == "__main__":
    unittest.main()
