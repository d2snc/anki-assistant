import os
import time
import threading
import logging
import sys
import subprocess
import re

import numpy as np
from faster_whisper import WhisperModel
from anki.collection import Collection
from html2text import html2text


def strip_images_from_text(html_text):
    """Remove tags <img> do HTML antes de converter para texto (TTS/avaliação)."""
    return re.sub(r'<img[^>]*>', '', html_text, flags=re.IGNORECASE)
import webview
import pyaudio
import torch
from silero_vad import load_silero_vad
import asyncio
import edge_tts

from config import ANKI_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig()


# Forçar Qt no pywebview (pula tentativa de GTK) e suprimir warning Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

TEST = len(sys.argv) > 1 and sys.argv[1] == "noaudio"

# Audio settings
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
num_samples = 512  # silero-vad pip exige 512 para 16000 Hz (ou 256 para 8000 Hz)
# Suprimir mensagens ALSA/JACK durante inicialização do PyAudio
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stderr = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)
audio = pyaudio.PyAudio()
os.dup2(_old_stderr, 2)  # Restaura stderr
os.close(_old_stderr)

# --- Configuração do LLM ---
# Opção 1 (padrão): Ollama local — instale com: curl -fsSL https://ollama.com/install.sh | sh
#                   depois: ollama pull llama3.2:3b
# Opção 2: Gemini API — descomente as linhas abaixo e comente o bloco ollama
import ollama as ollama_client
OLLAMA_MODEL = "qwen2.5:3b"

# Para usar Gemini em vez de Ollama, descomente:
# from google import genai as _genai
# _gemini_client = _genai.Client(api_key=GEMINI_KEY)
# GEMINI_MODEL = "gemini-2.0-flash-lite"

# Voz neural em PT-BR (Microsoft Edge TTS)
# Opções: pt-BR-FranciscaNeural (formal), pt-BR-ThalitaNeural (casual/jovem)
TTS_VOICE = "pt-BR-ThalitaNeural"


if not TEST:
    # Takes about 0.5 seconds, tiny.en is about 1.5s on slower machines
    model = WhisperModel(
        "medium",  # int8 é ~2x mais rápido que float32 com qualidade similar
        device="cpu",
        compute_type="int8",
    )

    # Para detecção de voz (VAD) — usa pacote pip silero-vad
    vad_model = load_silero_vad()


def confidence(chunk):
    """
    Usa Silero VAD para detectar se o usuário está falando.
    """
    audio_int16 = np.frombuffer(chunk, np.int16)
    abs_max = np.abs(audio_int16).max()
    audio_float32 = audio_int16.astype("float32")
    if abs_max > 0:
        audio_float32 *= 1 / 32768
    audio_float32 = audio_float32.squeeze()
    return vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()


def transcribe(audio_data):
    """
    Usa Whisper para transcrever áudio.
    """
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    # Prompt com vocabulário naval/marítimo para guiar o modelo
    MARITIME_PROMPT = (
        "rebocador, rebocadores, propulsão, propulsores, azimutal, azimutais, "
        "cicloidal, cicloidais, manobra, manobras, casco, proa, popa, costado, "
        "convés, calado, bolina, reboque, cabo de reboque, "
        "tubulão, kort, manobra portuária, força de tiro"
    )
    segments, _ = model.transcribe(
        audio_data / np.max(audio_data),
        language="pt",
        beam_size=7,
        without_timestamps=True,
        initial_prompt=MARITIME_PROMPT,
        condition_on_previous_text=False,
    )
    return "".join(x.text for x in segments)


def transcribe_answer():
    """
    Captura áudio do microfone e transcreve.

    Algoritmo de escuta:
    - Ouve continuamente após o usuário começar a falar
    - Se o usuário parar por 0.8s, transcreve o trecho
    - Se não falar por 2s, finaliza a transcrição.
    """
    if TEST:
        return input("Sua resposta: ")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    log.debug("Ouvindo...")

    prev_confidence = []
    data = []
    transcription = ""

    stop = threading.Event()
    last_spoken = [time.time()]

    def threaded_listen():
        while not stop.is_set():
            audio_chunk = stream.read(num_samples)
            chunk_confidence = confidence(audio_chunk)
            prev_confidence.append(chunk_confidence)

            mid_phrase = np.sum(prev_confidence[-5:]) > 5 * 0.7
            currently_speaking = chunk_confidence > 0.75

            if mid_phrase or currently_speaking:
                data.append(audio_chunk)

            if currently_speaking:
                last_spoken[0] = time.time()

    threading.Thread(target=threaded_listen, daemon=True).start()

    while not len(data):
        # Aguarda usuário começar a falar
        time.sleep(0.1)

    while True:
        speaking_gap = time.time() - last_spoken[0]

        if speaking_gap < 0.8:
            time.sleep(0.8 - speaking_gap)
        elif speaking_gap < 2.0 and len(data):
            log.debug(f"Transcrevendo... gap={speaking_gap:.1f}s, chunks={len(data)}")
            stt_start = time.time()
            next_chunk = b"".join(data)
            data.clear()
            transcription += transcribe(next_chunk)
            log.debug(f"STT em {time.time() - stt_start:.2f}s")
        else:  # speaking_gap > 2.0
            log.info("Transcrição finalizada")
            stop.set()
            data.clear()  # Descarta áudio acumulado durante transcrição longa
            log.debug(transcription)
            return transcription

    stop.set()


def tts(text):
    """
    Converte texto em fala com streaming Edge TTS.
    O áudio começa a tocar em ~300ms sem esperar o arquivo completo.
    """
    if TEST:
        return log.debug(f"[TTS] {text}")

    async def _stream():
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             "-probesize", "32", "-analyzeduration", "0", "-i", "pipe:0"],
            stdin=subprocess.PIPE,
        )
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    proc.stdin.write(chunk["data"])
        finally:
            proc.stdin.close()
            proc.wait()

    asyncio.run(_stream())


def make_latex_speakable(text):
    """
    Usa LLM local (Ollama) para converter LaTeX/símbolos em texto falável.
    """
    if "\\" not in text and "$" not in text:
        return text
    response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt="Traduza todo LaTeX/símbolos para que possa ser lido em voz alta naturalmente. "
               "Retorne APENAS o texto traduzido, sem mais nada:\n" + text,
    )
    return response.response.strip()


def evaluate_response(question, answer, user_response):
    """
    Avalia semanticamente a resposta do aluno.
    Retorna (score, feedback) onde feedback é falado em caso de erro.
    Avalia por CONCEITO, não correspondência exata de texto.
    """
    prompt = (
        "Você é um tutor de flashcards. Avalie se a resposta do aluno demonstra compreensão do conceito.\n\n"
        "IMPORTANTE: Avalie pelo CONCEITO, não pela correspondência exata de palavras. "
        "Se o aluno explicou a mesma ideia com outras palavras, considere correto.\n\n"
        "Notas:\n"
        "1 - Não sabe. Totalmente errado, em branco ou incoerente.\n"
        "2 - Demonstra algum conhecimento mas está incompleto ou parcialmente errado.\n"
        "3 - Parcialmente correto — acertou parte do conceito.\n"
        "4 - Correto — demonstra compreensão, mesmo que em outras palavras.\n\n"
        "Se a nota for menor que 4, explique em 1 ou 2 frases (em português) o que estava errado ou faltando.\n\n"
        "Responda EXATAMENTE neste formato (sem nada antes ou depois):\n"
        "NOTA: <dígito>\n"
        "FEEDBACK: <explicação, ou deixe em branco se nota 4>\n\n"
        f"Pergunta: {question}\n"
        f"Resposta correta: {answer}\n"
        f'Resposta do aluno: "{user_response}"'
    )
    response = ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
    raw = response.response.strip()
    log.debug(f"LLM raw: {raw!r}")

    score = 2
    feedback = ""
    for line in raw.splitlines():
        if line.upper().startswith("NOTA:"):
            for char in line:
                if char in "1234":
                    score = int(char)
                    break
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return score, feedback


def main_backend(window):
    collection = Collection(ANKI_PATH)
    media_dir = collection.media.dir()  # Ex: ~/.local/share/Anki2/.../collection.media

    def display_html(html):
        # Injeta <base> para que imagens relativas do Anki sejam encontradas
        base_tag = f'<base href="file://{media_dir}/">'
        html_with_base = base_tag + html
        window.evaluate_js(f"window.updateHtml(String.raw`{html_with_base}`);")

    try:
        while current_card := collection.sched.getCard():
            # TODO: handle cloze cards
            if "basic" not in current_card.note_type()["name"].lower():
                log.debug("Pulando card cloze")
                collection.sched.bury_cards([current_card.id])
                continue

            (question, answer_html) = current_card.note().fields

            # Pular cards cuja RESPOSTA contenha imagens
            if re.search(r'<img', answer_html, re.IGNORECASE):
                log.debug("Pulando card com imagem na resposta")
                collection.sched.bury_cards([current_card.id])
                continue

            # Texto limpo para TTS e avaliação (sem tags de imagem da pergunta)
            answer_text = html2text(strip_images_from_text(answer_html)).strip()

            # Pular cards com LaTeX renderizado (imagens de fórmulas)
            if "latex" in answer_text.lower():
                log.debug("Pulando card com LaTeX renderizado")
                collection.sched.bury_cards([current_card.id])
                continue

            display_html(current_card.render_output(browser=True).question_and_style())
            tts(make_latex_speakable(question))

            current_card.timer_started = time.time()  # Inicia timer para pontuação
            user_response = transcribe_answer()

            if "skip card" in user_response.lower():
                collection.sched.bury_cards([current_card.id])
                continue

            if "nao sei" in user_response.lower() or "não sei" in user_response.lower():
                score, feedback = 1, ""
            else:
                score, feedback = evaluate_response(question, answer_text, user_response)

            # Pisca a tela de vermelho ou verde dependendo da pontuação
            window.evaluate_js(
                f"window.flashScreen('{'#CC020255' if score < 4 else '#02CC0255'}');"
            )
            log.info(f"Score: {score} | Feedback: {feedback!r}")
            collection.sched.answerCard(current_card, score)

            if score == 4:
                # Acertou: elogiar e dar reforço com o feedback do LLM
                elogio = "Muito bom, acertou!"
                if feedback:
                    tts(f"{elogio} {feedback}")
                else:
                    tts(elogio)
            else:
                # Errou: uma fala natural combinando feedback + resposta correta
                correcao = f"Você errou. {feedback} A resposta correta é: {answer_text}" if feedback else f"Você errou. A resposta correta é: {answer_text}"
                display_html(
                    current_card.render_output(browser=True).answer_and_style()
                )
                tts(correcao)
                time.sleep(2)
    finally:
        collection.close()


if __name__ == "__main__":
    window = webview.create_window(
        "Anki Voice Assistant",
        html=open("display_card.html", "r").read(),
    )
    webview.start(main_backend, window, gui="qt")
