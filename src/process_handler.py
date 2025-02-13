import asyncio
from asyncio import Queue as AsyncQueue
import json
import sys

class ProcessHandler:
    def __init__(self):
        self._process = None
        self._output_queue = AsyncQueue()
        self._is_running = False
        self._is_starting = False
        self._stream_tasks = []

    async def run(self, command: list, cwd: str):
        if self._is_running or self._is_starting:
            await self._output_queue.put(json.dumps({"status": "error", "message": "Process already running"}))
            return

        try:
            self._is_starting = True

            # Clear queue before run
            while not self._output_queue.empty():
                self._output_queue.get_nowait()

            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._is_running = True
            self._is_starting = False

            async def stream_output(stream, stream_type):
                while True:
                    line = await stream.readline()
                    if line:
                        message = json.dumps({
                            "stream": stream_type,
                            "message": line.decode().strip().replace("\n", "\\n")
                        })
                        print(message, flush=True)
                        if stream_type == "stdout":
                            await self._output_queue.put(message)
                    else:
                        break

            stdout_task = asyncio.create_task(stream_output(self._process.stdout, "stdout"))
            stderr_task = asyncio.create_task(stream_output(self._process.stderr, "stderr"))
            self._stream_tasks = [stdout_task, stderr_task]

            await self._process.wait()

            result_message = {
                "status": "success" if self._process.returncode == 0 else "error",
                "message": "Process completed successfully" if self._process.returncode == 0 else f"Process exited with code {self._process.returncode}"
            }
            await self._output_queue.put(json.dumps(result_message))

        except Exception as e:
            await self._output_queue.put(json.dumps({"status": "error", "message": str(e)}))

        finally:
            for task in self._stream_tasks:
                task.cancel()
            try:
                await asyncio.gather(*self._stream_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass

            self._stream_tasks = []
            self._is_starting = False
            self._process = None
            self._is_running = False

    async def status(self):
        return json.dumps({
            "is_running": (self._is_running or self._is_starting) and self._process is not None
        })

    def subscribe(self):
        return self._output_queue

    async def get_stream(self):
        while True:
            try:
                output = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield output
            except asyncio.TimeoutError:
                if not self._is_running and self._process is None:
                    break
                continue
