from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from acp_sdk import Metadata
from functools import reduce
import os, sys
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# decide which graph to import build_graph from
def import_builder(provider: str):
    if provider == "aws":
        from agent.graph_aws import build_graph
    elif provider == "wx":
        from agent.graph_wx import build_graph
    return build_graph

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("aws", "wx"):
        print("Usage: uv run server.py [aws|wx]")
        sys.exit(1)
    provider = sys.argv[1]
    
    build_graph = import_builder(provider)
    memory = MemorySaver()
    graph = build_graph()
    graph = graph.compile(checkpointer=memory)
    
    server = Server()
    @server.agent(
        metadata=Metadata(
            name="Business Case Assistant",
            framework="LangChain",
            ui={"type":"chat",
                "user_greeting": """To get started, could you please share the key points or objectives of your business case?"""},
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
            
    server.run()

if __name__ == "__main__":
    main()

