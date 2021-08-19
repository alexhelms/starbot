import asyncio
import concurrent.futures


class ProcessPool:
    def __init__(self):
        self._executor = concurrent.futures.ProcessPoolExecutor()

    async def submit(self, fn, *args):
        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(self._executor, fn, *args)
        return await fut
