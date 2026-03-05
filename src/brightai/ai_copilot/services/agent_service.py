from strands import Agent
from strands.models import BedrockModel
from strands.tools import tool
import structlog.stdlib

from brightai.ai_copilot.core.config import AgentSettings, ApplicationConfig
from brightai.ai_copilot.core.service_manager import get_service_manager

log = structlog.stdlib.get_logger()


class AgentService:
    """Service for initializing and interacting with Strands SDK AI Agents"""

    def __init__(self, config: AgentSettings, app_config: ApplicationConfig) -> None:
        self.config = config
        self.app_config = app_config
        self._agent: Agent | None = None

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def _create_retrieval_tool(self):
        """Create a retrieval tool for the agent to use."""

        @tool
        def search_documentation(query: str) -> str:
            """
            Search the engineering documentation for relevant information.

            Args:
                query: The search query or question to find relevant documentation for

            Returns:
                Relevant documentation snippets that help answer the query
            """
            # Get RAG service from service manager
            service_manager = get_service_manager()
            rag_service = service_manager.get_service_sync("rag")

            if rag_service is None or not rag_service.is_ready:
                return "Documentation search is currently unavailable. Please answer based on your general knowledge."

            try:
                # Retrieve relevant chunks
                chunks = rag_service.retrieve(query)

                if not chunks:
                    return "No relevant documentation found for this query."

                # Format context
                context = rag_service.format_context(chunks)
                log.info(f"Retrieved {len(chunks)} chunks for agent tool call")
                return context

            except Exception as e:
                log.error(f"Error during retrieval: {e}")
                log.exception("Retrieval error details:")
                return f"Error searching documentation: {str(e)}"

        return search_documentation

    def _create_agent(self) -> Agent:
        log.info(f"Creating Agent with Bedrock Model ID {self.config.bedrock_model_id}")
        bedrock_model = BedrockModel(model_id=self.config.bedrock_model_id)

        # Build tools list
        tools = []
        if self.app_config.rag.enabled:
            log.info("RAG is enabled, adding search_documentation tool to agent")
            tools.append(self._create_retrieval_tool())

        agent = Agent(
            name=self.config.name,
            description=self.config.description,
            model=bedrock_model,
            system_prompt=self.config.system_prompt,
            tools=tools,
        )
        return agent
