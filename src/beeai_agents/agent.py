from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from acp_sdk import Metadata
from functools import reduce
import os, sys
from langgraph.checkpoint.memory import MemorySaver
from beeai_agents.graph import build_graph

memory = MemorySaver()
graph = build_graph()
graph = graph.compile(checkpointer=memory)

server = Server()
@server.agent(
    name="Business Case Assistant",
    description="The Business Case Assistant uses advanced language models \
        and LangGraph to interview the user about project requirements and create \
        a business case document based on the user's responses.",
    metadata=Metadata(
        version="1.0.0",
        framework="LangGraph",
        programming_language="Python",
        ui={"type":"chat",
            "user_greeting": """To get started, could you please share the key points or objectives of your business case?"""},
        author={
            "name": "Caitlin Tran",
        },
        recommended_models=[
            "claude-3-7-sonnet-latest",
        ],
        env=[
            {"name": "LLM_MODEL", "description": "Model to use from the specified OpenAI-compatible API."},
            {"name": "LLM_API_BASE", "description": "Base URL for OpenAI-compatible API endpoint"},
            {"name": "LLM_API_KEY", "description": "API key for OpenAI-compatible API endpoint"},
            {"name": "LLM_PROVIDER", "description": "The provider the LLM is from"}
        ],
    )
)
async def business_case_assistant(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    message = input[-1].parts[0].content
    config = {"configurable": {"thread_id": context.session_id}}

    async for event in graph.astream({"messages": str(message)}, config, stream_mode="updates"):
        output = event
        print(output)
        node = list(output.keys())[0]
        if node == 'Gathering Requirements':
            output = output.get('Gathering Requirements', {}).get('messages', [])
            for msg in output:
                yield MessagePart(content=msg.content)
        elif node == 'Compiling Document':
            output = output.get('Compiling Document', {}).get('document', [])
            yield MessagePart(content = output)
        
def run():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))

if __name__ == "__main__":
    run()
    

