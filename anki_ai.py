import os
import json
import base64
import time
import threading
import logging
import sys
import subprocess
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import requests
from anki.collection import Collection
from anki.cards import Card
from html2text import html2text
import io
import wave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Verifica qual API será usada
if os.getenv("GROQ_API_KEY"):
    openai_client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    LLM_MODEL = "llama-3.1-8b-instant"
    WHISPER_MODEL = "whisper-large-v3"
elif os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    LLM_MODEL = "gpt-4o-mini"
    WHISPER_MODEL = "whisper-1"
else:
    print("\nERRO: Nenhuma chave de API encontrada (GROQ_API_KEY ou OPENAI_API_KEY).")
    print("Crie um arquivo .env usando o .env.example como modelo.\n")
    sys.exit(1)

# Cliente exclusivo para OpenAI TTS (só inicializa se houver chave da OpenAI)
openai_tts_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_tts_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Transcrição (speech-to-text) via OpenRouter. O endpoint /audio/transcriptions
# do OpenRouter NÃO é compatível com o SDK da OpenAI (espera JSON com o áudio em
# base64, não multipart) e aceita só a dica de `language` — sem o prompt de
# vocabulário marítimo. Por isso é uma chamada HTTP direta. Sem OPENROUTER_API_KEY,
# cai no Whisper do cliente principal (Groq/OpenAI), que ainda usa esse prompt.
OPENROUTER_TRANSCRIBE_MODEL = "openai/gpt-4o-mini-transcribe"
OPENROUTER_TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

# Geração de imagem (Nano Banana) via OpenRouter para ilustrar flashcards. O
# modelo de imagem responde pelo endpoint de chat: pede-se modalities com
# "image" e a imagem volta como data URL base64 em
# choices[0].message.images[0].image_url.url.
OPENROUTER_IMAGE_MODEL = "google/gemini-2.5-flash-image"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Geração de VÍDEO (Veo 3.1 Fast) via OpenRouter para ilustrar flashcards. Ao
# contrário da imagem, o vídeo usa o endpoint assíncrono /videos: faz-se um POST
# que devolve um job com `polling_url`; aí faz-se polling até `completed` e baixa
# o MP4. Veja generate_illustration_video.
OPENROUTER_VIDEO_MODEL = "google/veo-3.1-fast"
OPENROUTER_VIDEO_URL = "https://openrouter.ai/api/v1/videos"

# Cliente OpenRouter (modelos grátis) usado pelo revisor de flashcards em
# background. OpenRouter é compatível com a API da OpenAI — basta trocar a
# base_url. Se a chave não estiver no .env, o revisor cai no Groq/OpenAI.
openrouter_client = None
if os.getenv("OPENROUTER_API_KEY"):
    openrouter_client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

# Modelos grátis do OpenRouter, tentados em ordem (os maiores primeiro). O
# último recurso é o cliente principal (Groq/OpenAI). Free tiers têm rate limit
# baixo — daí a lista e o fallback, para o worker não travar quando um recusa.
OPENROUTER_REVIEW_MODELS = [
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nex-agi/nex-n2-pro:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

# Modelo usado para JULGAR a resposta do aluno (evaluate_response). O Claude
# Sonnet 4.6 via OpenRouter avalia por conceito com bem mais consistência que os
# modelos free, sem dar nota errada em resposta claramente correta. É um modelo
# PAGO no OpenRouter — se a chamada falhar, o _llm_chat cai no Groq/OpenAI.
OPENROUTER_JUDGE_MODELS = [
    "anthropic/claude-sonnet-4.6",
]

# Modelo usado para TRANSFORMAR um flashcard difícil numa questão de múltipla
# escolha (generate_mcq). Claude Sonnet 4.6 via OpenRouter: precisa entender o
# conteúdo e criar alternativas plausíveis com um único gabarito. É PAGO — se a
# chamada falhar, o _llm_chat cai no Groq/OpenAI (qualidade menor, mas não trava).
OPENROUTER_MCQ_MODELS = [
    "anthropic/claude-sonnet-4.6",
]


def strip_images_from_text(html_text):
    """Remove tags <img> e tags de áudio do Anki antes de converter para texto (TTS/avaliação)."""
    text = re.sub(r'<img[^>]*>', '', html_text, flags=re.IGNORECASE)
    text = re.sub(r'\[anki:play:[^\]]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sound:[^\]]+\]', '', text, flags=re.IGNORECASE)
    return text

import torch
from silero_vad import load_silero_vad
import asyncio
import edge_tts

from config import ANKI_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig()


# Vozes neurais (Microsoft Edge TTS)
TTS_VOICE = "pt-BR-ThalitaNeural"
TTS_VOICE_EN = "en-US-JennyNeural"
USE_OPENAI_TTS = os.getenv("OPENAI_TTS", "").lower() in {"1", "true", "yes", "sim"}

_PT_CHARS = frozenset("ãâáàêéôóúüçíõÃÂÁÀÊÉÔÓÚÜÇÍÕ")

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # resultados determinísticos

    def is_english(*texts: str) -> bool:
        """True se o conjunto de textos (pergunta + resposta) for majoritariamente
        inglês. Combina tudo e deixa o langdetect decidir pela língua predominante,
        de modo que a voz acompanhe a maioria do conteúdo, não um trecho isolado."""
        combined = " ".join(t for t in texts if t).strip()
        if not combined:
            return False
        try:
            return _langdetect(combined) == "en"
        except Exception:
            # Sem detecção possível: qualquer acento PT indica português.
            return not any(c in _PT_CHARS for c in combined)

except ImportError:
    def is_english(*texts: str) -> bool:
        """Fallback sem langdetect: decide pela maioria entre palavras-função
        inglesas e caracteres acentuados do português no texto combinado."""
        combined = " ".join(t for t in texts if t)
        if not combined.strip():
            return False
        _EN = frozenset({"the","is","are","was","were","what","which","how","why",
                         "when","where","who","that","this","with","from","for",
                         "and","but","not","have","has","had","will","would",
                         "should","could","can","may","might","of","in","at",
                         "by","on","be","been","do","does","did"})
        en_hits = sum(1 for w in re.findall(r'[a-z]+', combined.lower()) if w in _EN)
        pt_hits = sum(1 for c in combined if c in _PT_CHARS)
        return en_hits > pt_hits

# Audio settings (usados por desktop e web)
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
num_samples = 512  # silero-vad pip exige 512 para 16000 Hz

# Modelos carregados sob demanda (evita carregar ao importar)
whisper_model = None
vad_model = None


def ensure_models_loaded():
    """Carrega VAD se ainda não carregado."""
    global vad_model
    if vad_model is None:
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
    return vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()  # type: ignore


_MARITIME_PROMPT_PT = (
    "rebocador, rebocadores, propulsão, propulsores, azimutal, azimutais, "
    "cicloidal, cicloidais, manobra, manobras, casco, proa, popa, costado, "
    "convés, calado, bolina, reboque, cabo de reboque, "
    "tubulão, kort, manobra portuária, força de tiro, "
    "GPS, DGPS, AIS, ECDIS, radar, carta 12000, carta náutica, escala"
)
_MARITIME_PROMPT_EN = (
    "tug, tugboat, propulsion, azimuth thruster, cycloidal propeller, "
    "mooring, berthing, hull, bow, stern, draft, freeboard, towing, towline, "
    "kort nozzle, bollard pull, GPS, DGPS, AIS, ECDIS, radar, nautical chart, "
    "gyrocompass, magnetic compass, bearing, heading, knots, nautical miles"
)


def _transcribe_openrouter(file_obj, lang):
    """POST direto no endpoint /audio/transcriptions do OpenRouter (JSON com o
    áudio WAV em base64). Devolve o texto transcrito, ou None se falhar (para o
    transcribe() cair no fallback Whisper). lang: 'pt' ou 'en'."""
    try:
        file_obj.seek(0)
        audio_b64 = base64.b64encode(file_obj.read()).decode("ascii")
        resp = requests.post(
            OPENROUTER_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={
                "model": OPENROUTER_TRANSCRIBE_MODEL,
                "input_audio": {"data": audio_b64, "format": "wav"},
                "language": lang,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("text")
        if text and text.strip():
            return text.strip()
        log.warning(f"Transcrição OpenRouter sem texto: {resp.text!r}")
    except Exception as e:
        log.warning(f"Transcrição OpenRouter falhou ({OPENROUTER_TRANSCRIBE_MODEL}): {e}")
    return None


def transcribe(file_obj, lang="pt"):
    """Transcreve áudio (WAV). lang: 'pt' ou 'en'.

    Usa o gpt-4o-mini-transcribe via OpenRouter quando há OPENROUTER_API_KEY;
    senão (ou se a chamada falhar) cai no Whisper do cliente principal (Groq/
    OpenAI), que ainda aproveita o prompt de vocabulário marítimo."""
    if os.getenv("OPENROUTER_API_KEY"):
        text = _transcribe_openrouter(file_obj, lang)
        if text is not None:
            return text
        file_obj.seek(0)  # rebobina para a tentativa de fallback

    prompt = _MARITIME_PROMPT_EN if lang == "en" else _MARITIME_PROMPT_PT
    resp = openai_client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=file_obj,
        language=lang,
        prompt=prompt,
    )
    return resp.text


def transcribe_answer(audio_interface=None, lang="pt"):
    """
    Captura áudio do microfone por 8 segundos fixos e transcreve.
    """
    import pyaudio

    if audio_interface is None:
        return input("Sua resposta: ")

    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    log.info("Gravando por 8 segundos...")

    frames = []
    num_iterations = int((SAMPLE_RATE / CHUNK) * 8)

    for _ in range(num_iterations):
        audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(audio_chunk)

    log.info("Gravação finalizada. Transcrevendo...")

    next_chunk = b"".join(frames)

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(next_chunk)
    wav_io.seek(0)
    wav_io.name = "audio.wav"

    transcription = transcribe(wav_io, lang=lang)
    log.debug(f"Transcrição finalizada")
    log.debug(transcription)

    stream.stop_stream()
    stream.close()

    return transcription


def tts(text, voice=None):
    """
    Converte texto em fala com streaming Edge TTS.
    O áudio começa a tocar em ~300ms sem esperar o arquivo completo.
    """
    if USE_OPENAI_TTS and openai_tts_client is not None:
        try:
            resp = openai_tts_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=resp.content)
            return
        except Exception as e:
            log.error(f"Erro no OpenAI TTS: {e}. Usando fallback (edge-tts)")

    async def _stream():
        communicate = edge_tts.Communicate(text, voice or TTS_VOICE)
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


async def tts_to_bytes(text, voice=None):
    """
    Gera áudio TTS e retorna como bytes MP3 (para envio via WebSocket).
    Usa OpenAI TTS se disponível (ultra-rápido), senão usa edge-tts.
    """
    if USE_OPENAI_TTS and openai_tts_client is not None:
        try:
            resp = openai_tts_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            return resp.content
        except Exception as e:
            log.error(f"Erro no OpenAI TTS: {e}. Usando fallback (edge-tts)")

    try:
        communicate = edge_tts.Communicate(text, voice or TTS_VOICE)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)
    except Exception as e:
        log.error(f"Erro no edge-tts: {e}. Seguindo sem áudio.")
        return b""


def strip_punctuation_for_tts(text):
    """
    Remove sinais de pontuação e artefatos Markdown que soam mal em TTS.
    Preserva letras, números, espaços e pontos de reticências naturais.
    """
    # Remove formatação Markdown (negrito, itálico, listas)
    text = re.sub(r'[*_~`#]+', '', text)
    # Remove pontuação que não agrega ao áudio
    text = re.sub(r'[\-–—•,;:!?()\[\]{}<>/\\|@#^&+=]', ' ', text)
    text = text.replace('\n', ' ')
    # Normaliza múltiplos espaços
    text = re.sub(r' +', ' ', text).strip()
    return text


LETTER_WORDS = {
    "a": "a",
    "b": "b",
    "be": "b",
    "c": "c",
    "ce": "c",
    "d": "d",
    "de": "d",
    "e": "e",
    "f": "f",
    "efe": "f",
    "g": "g",
    "ge": "g",
    "h": "h",
    "aga": "h",
    "i": "i",
    "j": "j",
    "jota": "j",
    "k": "k",
    "ka": "k",
    "l": "l",
    "ele": "l",
    "m": "m",
    "eme": "m",
    "n": "n",
    "ene": "n",
    "o": "o",
    "p": "p",
    "pe": "p",
    "q": "q",
    "que": "q",
    "r": "r",
    "erre": "r",
    "s": "s",
    "esse": "s",
    "t": "t",
    "te": "t",
    "u": "u",
    "v": "v",
    "ve": "v",
    "w": "w",
    "dabliu": "w",
    "x": "x",
    "xis": "x",
    "y": "y",
    "ipsilon": "y",
    "z": "z",
    "ze": "z",
}

NUMBER_WORDS = {
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
    "onze": 11,
    "doze": 12,
    "treze": 13,
    "quatorze": 14,
    "catorze": 14,
    "quinze": 15,
    "dezesseis": 16,
    "dezessete": 17,
    "dezoito": 18,
    "dezenove": 19,
    "vinte": 20,
    "trinta": 30,
    "quarenta": 40,
    "cinquenta": 50,
    "sessenta": 60,
    "setenta": 70,
    "oitenta": 80,
    "noventa": 90,
    "cem": 100,
    "cento": 100,
    "duzentos": 200,
    "trezentos": 300,
    "quatrocentos": 400,
    "quinhentos": 500,
    "seiscentos": 600,
    "setecentos": 700,
    "oitocentos": 800,
    "novecentos": 900,
}


def add_acronym_tokens(tokens):
    expanded = []
    for token in tokens:
        expanded.append(LETTER_WORDS.get(token, token))

    aliases = set(expanded)
    for start in range(len(expanded)):
        letters = []
        for token in expanded[start:start + 8]:
            if len(token) != 1 or not token.isalpha():
                break
            letters.append(token)
            if len(letters) >= 2:
                aliases.add("".join(letters))

    return expanded + sorted(aliases - set(expanded))


def number_phrase_to_int(words):
    total = 0
    current = 0
    used = False

    for word in words:
        if word == "e":
            continue
        if word == "mil":
            total += (current or 1) * 1000
            current = 0
            used = True
            continue
        value = NUMBER_WORDS.get(word)
        if value is None:
            return None
        current += value
        used = True

    if not used:
        return None

    return total + current


def add_number_tokens(tokens):
    aliases = set()
    number_vocab = set(NUMBER_WORDS) | {"e", "mil"}

    for start in range(len(tokens)):
        phrase = []
        for token in tokens[start:start + 8]:
            if token not in number_vocab:
                break
            phrase.append(token)
            value = number_phrase_to_int(phrase)
            if value is not None:
                aliases.add(str(value))

    return tokens + sorted(aliases)


def normalize_answer_text(text):
    """Normaliza texto para comparações óbvias sem depender do LLM."""
    text = html2text(strip_images_from_text(text))
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = re.sub(r'\s+', ' ', text).strip().split()
    tokens = add_acronym_tokens(tokens)
    tokens = add_number_tokens(tokens)
    return " ".join(tokens)


def is_obvious_match(answer, user_response):
    """
    Detecta acertos muito evidentes antes do LLM.
    Evita reprovar respostas curtas mas específicas, como "carta 12000".
    """
    normalized_answer = normalize_answer_text(answer)
    normalized_user = normalize_answer_text(user_response)

    if not normalized_answer or not normalized_user:
        return False

    if normalized_answer == normalized_user:
        return True

    answer_tokens = set(normalized_answer.split())
    user_tokens = set(normalized_user.split())
    significant_answer_tokens = {t for t in answer_tokens if len(t) > 2 or t.isdigit()}
    significant_user_tokens = {t for t in user_tokens if len(t) > 2 or t.isdigit()}
    user_numbers = set(re.findall(r'\d+', normalized_user))
    answer_numbers = set(re.findall(r'\d+', normalized_answer))

    if significant_answer_tokens and significant_answer_tokens <= significant_user_tokens:
        return True

    if user_numbers & answer_numbers and significant_user_tokens & answer_tokens:
        return True

    shorter, longer = sorted([normalized_answer, normalized_user], key=len)
    if len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.65:
        return True

    return SequenceMatcher(None, normalized_answer, normalized_user).ratio() >= 0.86


# Nota mínima do avaliador (1–4) para considerar o card ACERTADO. Acerto mantém
# o ease do avaliador (Good/Easy) e respeita o intervalo do Anki; abaixo disso o
# card é respondido como Again (1) e entra em reaprendizado, seguindo o fluxo
# natural do Anki: volta a aparecer após o passo de aprendizado e só "passa de
# dia" quando acertado. Nenhum card é enterrado nesse processo.
PASS_SCORE = 3


def get_prioritized_card(collection, fetch_limit=1, predicate=None):
    """
    Retorna o próximo card seguindo a ordem nativa do Anki, que já intercala
    cards novos, de aprendizado e de revisão conforme as opções do deck
    (New/review order: Mix with reviews / Show before / Show after reviews).
    Assim, cards novos também aparecem durante a sessão, e não só depois de
    zerar todas as revisões.

    Equivalente ao scheduler do próprio Anki (Scheduler.getCard): pega o
    primeiro card da fila já ordenada pelo backend.

    Se `predicate` for fornecido, percorre a janela dos próximos `fetch_limit`
    cards e retorna o primeiro que satisfaz predicate(card), ignorando os demais
    sem alterar o agendamento deles (útil para filtrar por tipo de note).
    """
    queued_cards = collection.sched.get_queued_cards(fetch_limit=fetch_limit)
    for queued in queued_cards.cards:
        card = Card(collection)
        card._load_from_backend_card(queued.card)
        if predicate is not None and not predicate(card):
            continue
        card.start_timer()
        return card
    return None


def make_latex_speakable(text):
    """
    Usa LLM (OpenAI/Groq) para converter LaTeX/símbolos em texto falável.
    """
    if "\\" not in text and "$" not in text:
        return text
    
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": "Traduza todo LaTeX/símbolos para que possa ser lido em voz alta naturalmente. "
                       "Retorne APENAS o texto traduzido, sem mais nada:\n" + text
        }]
    )
    return resp.choices[0].message.content.strip()



def _llm_chat(prompt, models=None, temperature=0.2):
    """Chama um LLM tentando os modelos free do OpenRouter em ordem e caindo no
    cliente principal (Groq/OpenAI) como último recurso. Devolve o texto da
    resposta ou None se tudo falhar. `models` permite usar uma lista própria;
    por padrão usa os modelos do OpenRouter."""
    models = OPENROUTER_REVIEW_MODELS if models is None else models
    attempts = []
    if openrouter_client is not None:
        attempts.extend((openrouter_client, m) for m in models)
    attempts.append((openai_client, LLM_MODEL))

    for client, model in attempts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            log.warning(f"LLM: modelo {model} falhou: {e}")
            continue
    return None


def generate_illustration(question, answer, book=None):
    """Gera uma ILUSTRAÇÃO DIDÁTICA (PNG) para um flashcard usando o Nano Banana
    (google/gemini-2.5-flash-image) via OpenRouter. O contexto do livro/assunto
    (nome do deck) é passado para a imagem combinar com a fonte do card — ex.:
    um card do livro "Arte Naval" gera uma figura coerente com o tema.

    Devolve os bytes PNG ou None (sem OPENROUTER_API_KEY ou falha na geração).
    A imagem volta como data URL base64 em message.images[0].image_url.url."""
    if not os.getenv("OPENROUTER_API_KEY"):
        log.warning("Geração de imagem indisponível: sem OPENROUTER_API_KEY")
        return None

    contexto = f'Este flashcard faz parte do material de estudo "{book}". ' if book else ""
    prompt = (
        "Crie uma ILUSTRAÇÃO DIDÁTICA para ajudar a memorizar um flashcard de estudo. "
        f"{contexto}"
        "A imagem deve representar visualmente o CONCEITO da resposta, como uma figura "
        "de livro didático: clara, simples e fiel ao conteúdo. NÃO escreva textos, "
        "palavras, legendas ou números na imagem. Use um fundo limpo.\n\n"
        f"Pergunta do flashcard: {question}\n"
        f"Resposta do flashcard: {answer}"
    )
    try:
        resp = requests.post(
            OPENROUTER_CHAT_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={
                "model": OPENROUTER_IMAGE_MODEL,
                "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        images = resp.json()["choices"][0]["message"].get("images") or []
        if not images:
            log.warning("Geração de imagem sem retorno de imagem")
            return None
        url = images[0]["image_url"]["url"]
        b64 = url.split(",", 1)[1] if "," in url else url
        return base64.b64decode(b64)
    except Exception as e:
        log.warning(f"Geração de imagem falhou ({OPENROUTER_IMAGE_MODEL}): {e}")
        return None


def _find_video_urls(obj):
    """Vasculha recursivamente o JSON de um job de vídeo do OpenRouter atrás de
    URLs de download já prontas (não precisam de auth). O formato da resposta
    varia entre modelos, então em vez de fixar um caminho (unsigned_urls, output.
    video, assets...), coletamos toda string http(s) e priorizamos as que parecem
    ser o arquivo de vídeo (.mp4/.webm ou chave 'url'/'unsigned')."""
    found = []

    def walk(node, key=""):
        if isinstance(node, str):
            if node.startswith("http"):
                score = 0
                low = node.lower()
                k = key.lower()
                if ".mp4" in low or ".webm" in low or ".mov" in low:
                    score += 2
                if "unsigned" in k or k in ("url", "video", "video_url"):
                    score += 1
                # Evita a própria polling_url / links de API (não são o arquivo).
                if "/api/v1/videos" in low:
                    score -= 3
                found.append((score, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, key)

    walk(obj)
    found.sort(key=lambda t: t[0], reverse=True)
    return [u for score, u in found if score >= 0]


def generate_illustration_video(question, answer, book=None):
    """Gera um VÍDEO DIDÁTICO curto (MP4) para um flashcard usando o Veo 3.1 Fast
    (google/veo-3.1-fast) via OpenRouter. O objetivo é ajudar a entender melhor a
    RESPOSTA do card — uma animação clara do conceito, com o contexto do livro/
    assunto (nome do deck) para casar com a fonte.

    A geração é assíncrona: um POST em /videos devolve um job com `polling_url`;
    faz-se polling até `completed` e então baixa-se o MP4. Devolve os bytes MP4 ou
    None (sem OPENROUTER_API_KEY, timeout do polling ou falha na geração)."""
    if not os.getenv("OPENROUTER_API_KEY"):
        log.warning("Geração de vídeo indisponível: sem OPENROUTER_API_KEY")
        return None

    contexto = f'Este flashcard faz parte do material de estudo "{book}". ' if book else ""
    prompt = (
        "Crie um VÍDEO DIDÁTICO curto para ajudar a ENTENDER e memorizar a resposta "
        "de um flashcard de estudo. "
        f"{contexto}"
        "O vídeo deve mostrar visualmente o CONCEITO da resposta, como uma animação "
        "de livro didático: clara, simples e fiel ao conteúdo, com movimento que "
        "ajude a compreender a ideia. Evite texto escrito na tela.\n\n"
        f"Pergunta do flashcard: {question}\n"
        f"Resposta do flashcard (o que o vídeo precisa explicar): {answer}"
    )
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    try:
        resp = requests.post(
            OPENROUTER_VIDEO_URL,
            headers=headers,
            json={
                "model": OPENROUTER_VIDEO_MODEL,
                "prompt": prompt,
                "duration": 4,
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "generate_audio": True,
            },
            timeout=60,
        )
        resp.raise_for_status()
        job = resp.json()
    except Exception as e:
        log.warning(f"Geração de vídeo falhou ao iniciar ({OPENROUTER_VIDEO_MODEL}): {e}")
        return None

    job_id = job.get("id")
    poll_url = job.get("polling_url") or (f"{OPENROUTER_VIDEO_URL}/{job_id}" if job_id else None)
    if not poll_url:
        log.warning(f"Geração de vídeo sem polling_url/id: {job}")
        return None

    # Vídeo leva ~1-3 min. Polling limitado para o worker não rodar pra sempre.
    for _ in range(40):
        time.sleep(8)
        try:
            pr = requests.get(poll_url, headers=headers, timeout=60)
            pr.raise_for_status()
            job = pr.json()
        except Exception as e:
            log.warning(f"Geração de vídeo: polling falhou: {e}")
            continue
        status = job.get("status")
        if status == "completed":
            break
        if status in ("failed", "cancelled", "expired"):
            log.warning(f"Geração de vídeo terminou como '{status}': {job.get('error')}")
            return None
    else:
        log.warning("Geração de vídeo: tempo esgotado no polling")
        return None

    # Baixa o MP4. O job completo traz as URLs prontas (assinadas/unsigned) em
    # algum lugar do JSON — o formato varia, então vasculhamos recursivamente
    # atrás de qualquer URL de vídeo, que baixa SEM auth. Só caímos no endpoint
    # autenticado /content?index=0 se não acharmos nenhuma (ele deu 401 antes).
    urls = _find_video_urls(job)
    log.info(f"Geração de vídeo: job concluído, chaves={list(job.keys())}, urls={urls}")
    try:
        if urls:
            vr = requests.get(urls[0], timeout=120)
        else:
            vr = requests.get(f"{OPENROUTER_VIDEO_URL}/{job_id}/content?index=0",
                              headers=headers, timeout=120)
        vr.raise_for_status()
        return vr.content
    except Exception as e:
        log.warning(f"Geração de vídeo: download falhou: {e}")
        return None


# Prompt para converter um flashcard pergunta-e-resposta numa questão de
# múltipla escolha. Objetivo: cartão difícil vira uma questão mais fácil de
# revisar. O enunciado SEMPRE cita a referência ("De acordo com o <livro>, ...")
# para ancorar a memória. O gabarito reproduz a resposta original; os distratores
# são plausíveis, porém errados. A saída é um JSON estrito (parseado abaixo).
MCQ_PROMPT = """Você transforma um flashcard de pergunta-e-resposta numa QUESTÃO DE MÚLTIPLA ESCOLHA, para tornar um cartão difícil mais fácil de revisar.

Fonte/referência do material (use o nome do livro/assunto principal, não o caminho todo): <<<BOOK>>>

Flashcard original:
Pergunta: <<<QUESTION>>>
Resposta correta: <<<ANSWER>>>

Regras:
- ENUNCIADO ("question"): reescreva a pergunta como um enunciado claro e, SEMPRE que houver uma referência acima, CITE-A no começo, no formato "De acordo com o <referência>, ...". Cite a fonte pelo NOME e de forma natural (ex.: "De acordo com o Arte Naval, ..."); NÃO repita o caminho do baralho, barras (/) nem "::". Se a referência tiver vários níveis, use só o nome principal do livro/assunto. Se não houver referência útil, faça um enunciado claro sem citar.
- ALTERNATIVAS ("options"): crie de 4 a 5 alternativas. EXATAMENTE UMA é a correta e deve reproduzir fielmente a Resposta correta do flashcard. As outras são plausíveis, do mesmo tema/tipo, mas claramente erradas para quem estudou o material. NÃO escreva a letra (A), B)...) dentro do texto — só o texto da alternativa.
- GABARITO ("answer"): a letra da alternativa correta (uma letra maiúscula).
- "note": explique em uma frase por que a correta está certa, usando só o conteúdo do flashcard.
- "notes": para CADA letra, uma frase dizendo por que aquela alternativa está certa ou errada.
- Mantenha o MESMO IDIOMA do flashcard. NÃO invente fatos que contrariem o flashcard.

Responda SOMENTE com um JSON válido, sem texto antes ou depois, neste formato EXATO:
{"question": "<enunciado>", "options": {"A": "<texto>", "B": "<texto>", "C": "<texto>", "D": "<texto>"}, "answer": "B", "note": "<por que a correta está certa>", "notes": {"A": "<por quê>", "B": "<por quê>", "C": "<por quê>", "D": "<por quê>"}}"""


def _parse_mcq_json(raw):
    """Extrai e valida o JSON da múltipla escolha. Devolve um dict normalizado
    {question, options{A..}, answer, note, notes{A..}} com as letras REORDENADAS
    de forma contígua a partir de 'A' (o gabarito é remapeado junto), ou None se
    a resposta do LLM for inválida."""
    text = raw.strip()
    text = re.sub(r'^```(?:json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    question = str(data.get("question", "")).strip()
    raw_options = data.get("options")
    raw_notes = data.get("notes") or {}
    answer = str(data.get("answer", "")).strip().upper()[:1]
    if not question or not raw_options:
        return None

    # options pode vir como dict {"A": "..."} ou como lista ["...", ...].
    if isinstance(raw_options, dict):
        ordered_keys = sorted(k.strip().upper() for k in raw_options.keys() if str(k).strip())
        texts = [str(raw_options[k]).strip() for k in ordered_keys]
        note_for = lambda i: str(raw_notes.get(ordered_keys[i], "")).strip() if isinstance(raw_notes, dict) else ""
        answer_idx = ordered_keys.index(answer) if answer in ordered_keys else -1
    elif isinstance(raw_options, list):
        texts = [str(o).strip() for o in raw_options]
        note_list = raw_notes if isinstance(raw_notes, list) else []
        note_for = lambda i: str(note_list[i]).strip() if i < len(note_list) else ""
        # Na forma de lista o gabarito costuma vir como letra (A=0, B=1, ...).
        answer_idx = ord(answer) - ord("A") if answer.isalpha() else -1
    else:
        return None

    texts = [t for t in texts if t]
    if not (2 <= len(texts) <= 8):
        return None
    if not (0 <= answer_idx < len(texts)):
        return None

    letters = [chr(ord("A") + i) for i in range(len(texts))]
    options = dict(zip(letters, texts))
    notes = {}
    for i, letter in enumerate(letters):
        n = note_for(i)
        if n:
            notes[letter] = n
    return {
        "question": question,
        "options": options,
        "answer": letters[answer_idx],
        "note": str(data.get("note", "")).strip(),
        "notes": notes,
    }


def generate_mcq(question, answer, book=None):
    """Gera uma questão de MÚLTIPLA ESCOLHA a partir de um flashcard (texto da
    pergunta + resposta), via Claude Sonnet 4.6 no OpenRouter. O `book` (nome do
    deck/assunto) vira a referência citada no enunciado. Devolve o dict
    normalizado de _parse_mcq_json ou None se o LLM não respondeu de forma
    utilizável (ex.: sem OPENROUTER_API_KEY e fallback também falhou)."""
    prompt = (
        MCQ_PROMPT
        .replace("<<<BOOK>>>", book or "(sem referência)")
        .replace("<<<QUESTION>>>", question or "")
        .replace("<<<ANSWER>>>", answer or "")
    )
    raw = _llm_chat(prompt, models=OPENROUTER_MCQ_MODELS, temperature=0.4)
    if not raw:
        return None
    return _parse_mcq_json(raw)


def evaluate_response(question, answer, user_response, lang="pt"):
    """
    Avalia semanticamente a resposta do aluno.
    Retorna (score, feedback) onde feedback é falado em caso de erro.
    Avalia por CONCEITO, não correspondência exata de texto.
    lang: "pt" para português, "en" para inglês.

    O julgamento roda pelo Claude Sonnet 4.6 (OpenRouter), caindo no cliente
    principal (Groq/OpenAI) só como último recurso: um modelo fraco demais dá
    notas erradas em respostas claramente corretas.
    """
    if is_obvious_match(answer, user_response):
        return 4, ""

    feedback_lang = "in English" if lang == "en" else "em português"
    prompt = (
        "Você é uma avaliadora rigorosa, mas justa, de flashcards.\n\n"
        "IMPORTANTE: Avalie pelo CONCEITO, não pela correspondência exata de palavras. "
        "Se o aluno explicou a mesma ideia com outras palavras, conta como compatível. "
        "Estime quanto do CONTEÚDO ESSENCIAL da Resposta correta a resposta do aluno cobre, "
        "como uma PORCENTAGEM de compatibilidade (0% a 100%), e atribua a nota por essa porcentagem. "
        "Não invente exigências que não estão na Resposta correta nem penalize por ela ser mais completa.\n"
        "Considere equivalentes números falados/escritos, siglas faladas letra por letra, pequenas variações de transcrição, artigos omitidos e ordem diferente das palavras.\n\n"
        "RESTRIÇÃO ABSOLUTA: use APENAS a Pergunta e a Resposta correta fornecidas abaixo. "
        "Não acrescente fatos, exemplos, causas, consequências ou explicações externas que não estejam no flashcard.\n\n"
        "Notas por porcentagem de compatibilidade com a Resposta correta:\n"
        "4 - ~100% compatível: cobre praticamente todo o conteúdo essencial (correto e completo).\n"
        "3 - ~90% ou mais compatível: cobre quase todo o conteúdo essencial (correto, faltou pouco).\n"
        "2 - ~70% compatível: cobre boa parte, mas está incompleto ou parcialmente errado.\n"
        "1 - ~40% ou menos compatível, ou nada compatível: errou (não sabe, em branco ou incoerente).\n\n"
        f"Regras para o FEEDBACK (sempre {feedback_lang}):\n"
        "- Se a nota for 4, deixe o FEEDBACK em branco.\n"
        "- Se a nota for 1, 2 ou 3, diga apenas o que faltou comparando com a Resposta correta.\n"
        "- O FEEDBACK deve ser curto e só pode reformular trechos que já estão na Resposta correta.\n"
        "- Nunca invente uma explicação além do texto do flashcard.\n\n"
        "Responda EXATAMENTE neste formato (sem nada antes ou depois):\n"
        "NOTA: <dígito>\n"
        "FEEDBACK: <texto do feedback, ou vazio se nota 4>\n\n"
        f"Pergunta: {question}\n"
        f"Resposta correta: {answer}\n"
        f'Resposta do aluno: "{user_response}"'
    )
    
    raw = _llm_chat(prompt, models=OPENROUTER_JUDGE_MODELS, temperature=0)
    if not raw:
        # Sem resposta de nenhum modelo: trata como erro brando (Again) sem feedback.
        return 1, ""
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


# ---------------------------------------------------------------------------
# Revisor de flashcards (worker em background da versão web)
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """Você é um especialista em flashcards eficazes (princípio do conhecimento mínimo).
Avalie o flashcard abaixo e decida se ele precisa de melhoria.

Critérios:
- DIVIDIR ("split"): o cartão cobre informação demais (vários fatos ou itens numa só resposta) e ficaria melhor como vários cartões menores, cada um com um único fato.
- EDITAR ("edit"): a pergunta ou a resposta está confusa, ambígua, prolixa ou não dá para responder objetivamente; reescreva de forma concisa e clara, mantendo o MESMO conteúdo.
- OK ("ok"): o cartão já está bom (foco único, pergunta clara, resposta concisa).

Regras absolutas:
- NÃO invente fatos novos. Use apenas a informação já presente no cartão.
- Mantenha o idioma original do cartão.
- Ao dividir, distribua os fatos existentes entre os cartões; não duplique nem acrescente nada.
- Seja conservador: só sugira mudança quando houver ganho claro. Na dúvida, responda "ok".

Responda SOMENTE com um JSON válido, sem nenhum texto antes ou depois, neste formato exato:
{"verdict": "ok", "reason": "<motivo curto em português>", "cards": []}
- Se "edit", "cards" deve ter exatamente 1 item: [{"front": "<pergunta>", "back": "<resposta>"}].
- Se "split", "cards" deve ter 2 ou mais itens, cada um {"front": "...", "back": "..."}.

FLASHCARD:
Frente: <<<FRONT>>>
Verso: <<<BACK>>>"""


def _review_chat(prompt):
    """Chama um LLM grátis para revisar flashcards: tenta os modelos free do
    OpenRouter em ordem e cai no cliente principal (Groq/OpenAI) como último
    recurso. Devolve o texto da resposta ou None se tudo falhar."""
    return _llm_chat(prompt, temperature=0.2)


def _parse_review_json(raw):
    """Extrai e valida o JSON da resposta do revisor. Devolve dict normalizado
    {verdict, reason, cards} ou None se a resposta for inválida."""
    text = raw.strip()
    # Remove cercas de código (```json ... ```), comuns em respostas de LLM.
    text = re.sub(r'^```(?:json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    # Pega o objeto JSON mais externo.
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in {"ok", "edit", "split"}:
        return None

    cards = []
    for c in data.get("cards", []) or []:
        if not isinstance(c, dict):
            continue
        front = str(c.get("front", "")).strip()
        back = str(c.get("back", "")).strip()
        if front or back:
            cards.append({"front": front, "back": back})

    # Coerência entre o veredito e a quantidade de cartões propostos.
    if verdict == "ok":
        cards = []
    elif verdict == "edit" and len(cards) != 1:
        return None
    elif verdict == "split" and len(cards) < 2:
        return None

    return {"verdict": verdict, "reason": str(data.get("reason", "")).strip(), "cards": cards}


def review_flashcard(front, back):
    """Revisa um flashcard (texto da frente e do verso) e devolve um dict
    {verdict, reason, cards} ou None se o LLM não respondeu de forma utilizável."""
    prompt = REVIEW_PROMPT.replace("<<<FRONT>>>", front).replace("<<<BACK>>>", back)
    raw = _review_chat(prompt)
    if not raw:
        return None
    return _parse_review_json(raw)


def main_backend(window):
    """Loop principal da versão desktop (pywebview)."""
    collection = Collection(ANKI_PATH)
    media_dir = collection.media.dir()

    def display_html(html):
        base_tag = f'<base href="file://{media_dir}/">'
        html_with_base = base_tag + html
        window.evaluate_js(f"window.updateHtml(String.raw`{html_with_base}`);")

    try:
        while True:
            current_card = get_prioritized_card(collection)
            if current_card is None:
                break

            if "basic" not in current_card.note_type()["name"].lower():
                log.debug("Pulando card cloze")
                collection.sched.bury_cards([current_card.id])
                continue

            (question, answer_html) = current_card.note().fields

            if re.search(r'<img', answer_html, re.IGNORECASE):
                log.debug("Pulando card com imagem na resposta")
                collection.sched.bury_cards([current_card.id])
                continue

            answer_text = html2text(strip_images_from_text(answer_html)).strip()

            if "latex" in answer_text.lower():
                log.debug("Pulando card com LaTeX renderizado")
                collection.sched.bury_cards([current_card.id])
                continue

            display_html(current_card.render_output(browser=True).question_and_style())
            card_in_english = is_english(question, answer_text)
            lang = "en" if card_in_english else "pt"
            voice = TTS_VOICE_EN if card_in_english else None
            speakable = make_latex_speakable(question)
            speakable = strip_punctuation_for_tts(speakable)
            if not speakable:
                speakable = "Verifique a tela."
            tts(speakable, voice=voice)

            current_card.timer_started = time.time()
            user_response = transcribe_answer(_desktop_audio, lang=lang)

            if "skip card" in user_response.lower():
                collection.sched.bury_cards([current_card.id])
                continue

            if "nao sei" in user_response.lower() or "não sei" in user_response.lower():
                score, feedback = 1, ""
            else:
                score, feedback = evaluate_response(question, answer_text, user_response, lang=lang)

            # Acerto (>= PASS_SCORE) mantém o ease do avaliador (Good/Easy) e
            # respeita o intervalo; erro vira Again (1) para reaprender o card.
            passed = score >= PASS_SCORE
            ease = score if passed else 1

            window.evaluate_js(
                f"window.flashScreen('{'#02CC0255' if passed else '#CC020255'}');"
            )
            log.info(f"Score: {score} | Feedback: {feedback!r}")
            try:
                collection.sched.answerCard(current_card, ease)
            except Exception as e:
                log.error(f"Erro ao responder card: {e}")
                continue

            answer_spoken = strip_punctuation_for_tts(answer_text)
            if passed:
                if card_in_english:
                    elogio = f"Well done! The answer is: {answer_spoken}" if answer_spoken else "Well done!"
                else:
                    elogio = f"Muito bom, acertou! A resposta é: {answer_spoken}" if answer_spoken else "Muito bom, acertou!"
                tts(f"{elogio} {feedback}" if feedback else elogio, voice=voice)
            else:
                if card_in_english:
                    correcao = f"Wrong. {feedback} The correct answer is: {answer_spoken}" if feedback else f"Wrong. The correct answer is: {answer_spoken}"
                else:
                    correcao = f"Você errou. {feedback} A resposta correta é: {answer_spoken}" if feedback else f"Você errou. A resposta correta é: {answer_spoken}"
                display_html(current_card.render_output(browser=True).answer_and_style())
                tts(correcao, voice=voice)
                time.sleep(2)
    finally:
        collection.close()


if __name__ == "__main__":
    import webview
    import pyaudio

    # Forçar Qt no pywebview (pula tentativa de GTK) e suprimir warning Wayland
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # Suprimir mensagens ALSA/JACK durante inicialização do PyAudio
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _old_stderr = os.dup(2)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    _desktop_audio = pyaudio.PyAudio()
    os.dup2(_old_stderr, 2)
    os.close(_old_stderr)

    ensure_models_loaded()

    window = webview.create_window(
        "Anki Voice Assistant",
        html=open("display_card.html", "r").read(),
    )
    webview.start(main_backend, window, gui="qt")
