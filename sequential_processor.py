import asyncio
import subprocess
import time
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from flight_agent import process_flight_data_from_file

class SequentialThinker:
    def __init__(self, server_command, server_args):
        self.server_command = server_command
        self.server_args = server_args
        self.thought_number = 0
        self.total_thoughts = 0
        self.branch_id = None
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=env
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def think(self, thought: str, next_thought_needed: bool = True, **kwargs):
        """Sends a thought to the sequential thinking tool."""
        self.thought_number += 1
        payload = {
            "thought": thought,
            "nextThoughtNeeded": next_thought_needed,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            **kwargs
        }
        if self.branch_id:
            payload["branchId"] = self.branch_id

        print(f"Step {self.thought_number}: {thought}")
        # The tool name is 'sequentialthinking' in this specific server.
        response = await self.session.call_tool("sequentialthinking", payload)

        if response:
            print(f"Tool response: {response.content}")
        return response

async def main():
    """
    Main function to orchestrate the flight data processing.
    """
    server_command = "npx.cmd" if os.name == 'nt' else "npx"
    server_args = ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    
    try:
        async with SequentialThinker(server_command, server_args) as thinker:
            thinker.total_thoughts = 3

            await thinker.think(
                thought="Start of flight data processing.",
                totalThoughts=thinker.total_thoughts
            )

            data_dir = Path(__file__).parent / "data"
            output_dir = Path(__file__).parent / "output"
            lofts_file = data_dir / "lofts.csv"
            sections_file = Path(__file__).parent / "sections_kwisa.json"
            flight_data_file = data_dir / "sample_flight.txt"

            await thinker.think(
                thought=f"Processing flight data from {flight_data_file}."
            )

            process_flight_data_from_file(
                file_path=flight_data_file,
                lofts_file=lofts_file,
                sections_file=sections_file,
                output_dir=output_dir
            )

            await thinker.think(
                thought="End of flight data processing.",
                next_thought_needed=False
            )
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    import os
    asyncio.run(main())
