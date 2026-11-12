def main() -> None:
    import asyncio

    from .bot import run

    asyncio.run(run())
