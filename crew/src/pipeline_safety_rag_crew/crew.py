"""
Pipeline Safety RAG Crew
========================
Two-agent CrewAI crew that answers PHMSA pipeline safety regulation
questions using the FAISS index already in this repository.

Agents
------
retrieval_specialist  — Queries the regulation index via RAGSearchTool.
regulatory_analyst    — Synthesises retrieved passages into a cited answer.

The LLM used by both agents is controlled by the MODEL env var
(default: ``bedrock/openai.gpt-oss-120b-1:0`` — same model as the
Strands-based copilot in the main application).

Usage
-----
    from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

    result = PipelineSafetyRAGCrew().crew().kickoff(
        inputs={"question": "What are the pressure test requirements under §192.505?"}
    )
    print(result.raw)
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from pipeline_safety_rag_crew.tools.rag_tool import RAGSearchTool


@CrewBase
class PipelineSafetyRAGCrew:
    """Two-agent crew: retrieve regulatory passages → synthesise a cited answer."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def retrieval_specialist(self) -> Agent:
        """Searches the FAISS regulation index via RAGSearchTool."""
        return Agent(
            config=self.agents_config["retrieval_specialist"],  # type: ignore[index]
            tools=[RAGSearchTool()],
            verbose=True,
        )

    @agent
    def regulatory_analyst(self) -> Agent:
        """Synthesises retrieved passages into a clear, cited answer."""
        return Agent(
            config=self.agents_config["regulatory_analyst"],  # type: ignore[index]
            verbose=True,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def retrieval_task(self) -> Task:
        return Task(config=self.tasks_config["retrieval_task"])  # type: ignore[index]

    @task
    def synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_task"],  # type: ignore[index]
            output_file="output/answer.md",
        )

    # ── Crew ──────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
