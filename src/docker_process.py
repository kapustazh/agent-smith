import asyncio
import codecs

from pydantic import BaseModel


class ProcessResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False


async def capture(
    arguments: list[str], timeout: float, limit: int, data: bytes = b"",
) -> ProcessResult:
    process = await asyncio.create_subprocess_exec(
        *arguments, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    result = ProcessResult()
    remaining = limit
    outputs: list[list[str]] = [[], []]

    async def read(stream: asyncio.StreamReader, index: int) -> None:
        nonlocal remaining
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = await stream.read(4096)
            text = decoder.decode(chunk, final=not chunk)
            accepted = text[:remaining]
            if accepted:
                outputs[index].append(accepted)
            remaining -= len(accepted)
            result.truncated |= len(text) > len(accepted)
            if not chunk:
                break

    async def write() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(data)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    assert process.stdout is not None and process.stderr is not None
    tasks = [asyncio.create_task(read(process.stdout, 0)),
             asyncio.create_task(read(process.stderr, 1)),
             asyncio.create_task(write())]
    work = asyncio.gather(process.wait(), *tasks)
    try:
        await asyncio.wait_for(asyncio.shield(work), timeout)
        result.exit_code = process.returncode
    except TimeoutError:
        result.timed_out = True
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(work, return_exceptions=True)
    result.stdout, result.stderr = ("".join(parts) for parts in outputs)
    notices = []
    if result.timed_out:
        notices.append("Execution timed out.")
    if result.truncated:
        notices.append("Output truncated.")
    if result.exit_code == 137:
        notices.append("Exit 137; check the memory limit.")
    if notices:
        notice = "\n[" + " ".join(notices) + "]"
        if (len(result.stdout + result.stderr) > max(0, limit - len(notice))
                and not result.truncated):
            result.truncated = True
            notices.append("Output truncated.")
            notice = "\n[" + " ".join(notices) + "]"
        room = max(0, limit - len(notice))
        result.stdout = result.stdout[:room]
        result.stderr = (result.stderr[:room - len(result.stdout)] +
                         notice[:limit])
    return result
