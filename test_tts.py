import asyncio
import edge_tts

async def main():
    communicate = edge_tts.Communicate(" ", "pt-BR-ThalitaNeural")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    print(f"Got {len(chunks)} chunks")

asyncio.run(main())
