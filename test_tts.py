import asyncio

import anki_ai


class _FailingCommunicate:
    def __init__(self, *args, **kwargs):
        pass

    def stream(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise RuntimeError("simulated edge-tts failure")


async def test_tts_to_bytes_returns_empty_audio_when_edge_tts_fails():
    original_use_openai_tts = anki_ai.USE_OPENAI_TTS
    original_communicate = anki_ai.edge_tts.Communicate
    try:
        anki_ai.USE_OPENAI_TTS = False
        anki_ai.edge_tts.Communicate = _FailingCommunicate

        audio = await anki_ai.tts_to_bytes("teste", voice="pt-BR-ThalitaNeural")

        assert audio == b""
    finally:
        anki_ai.USE_OPENAI_TTS = original_use_openai_tts
        anki_ai.edge_tts.Communicate = original_communicate


if __name__ == "__main__":
    asyncio.run(test_tts_to_bytes_returns_empty_audio_when_edge_tts_fails())
    print("OK — fallback do edge-tts retorna áudio vazio quando falha")
