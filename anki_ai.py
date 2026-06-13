import os
import time
import threading
import logging
import sys
import subprocess
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
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

    def is_english(text: str) -> bool:
        """True se o texto for detectado como inglês pelo langdetect."""
        if any(c in _PT_CHARS for c in text):
            return False
        clean = text.strip()
        if not clean:
            return False
        try:
            return _langdetect(clean) == "en"
        except Exception:
            return False

except ImportError:
    def is_english(text: str) -> bool:
        """Fallback: detecta inglês por palavras função comuns."""
        if any(c in _PT_CHARS for c in text):
            return False
        _EN = frozenset({"the","is","are","was","were","what","which","how","why",
                         "when","where","who","that","this","with","from","for",
                         "and","but","not","have","has","had","will","would",
                         "should","could","can","may","might","of","in","at",
                         "by","on","be","been","do","does","did"})
        words = set(re.findall(r'[a-z]+', text.lower()))
        return bool(words & _EN)

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


def transcribe(file_obj, lang="pt"):
    """Usa OpenAI/Groq API para transcrever áudio. lang: 'pt' ou 'en'."""
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

    communicate = edge_tts.Communicate(text, voice or TTS_VOICE)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


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



def evaluate_response(question, answer, user_response, lang="pt"):
    """
    Avalia semanticamente a resposta do aluno.
    Retorna (score, feedback) onde feedback é falado em caso de erro.
    Avalia por CONCEITO, não correspondência exata de texto.
    lang: "pt" para português, "en" para inglês.
    """
    if is_obvious_match(answer, user_response):
        return 4, ""

    feedback_lang = "in English" if lang == "en" else "em português"
    prompt = (
        "Você é uma avaliadora rigorosa, mas justa, de flashcards.\n\n"
        "IMPORTANTE: Avalie pelo CONCEITO, não pela correspondência exata de palavras. "
        "Se o aluno explicou a mesma ideia com outras palavras, considere correto. "
        "Se a resposta do aluno for curta, mas contiver o dado essencial do flashcard, considere correto. "
        "Considere equivalentes números falados/escritos, siglas faladas letra por letra, pequenas variações de transcrição, artigos omitidos e ordem diferente das palavras.\n\n"
        "RESTRIÇÃO ABSOLUTA: use APENAS a Pergunta e a Resposta correta fornecidas abaixo. "
        "Não acrescente fatos, exemplos, causas, consequências ou explicações externas que não estejam no flashcard.\n\n"
        "Notas:\n"
        "1 - Não sabe. Totalmente errado, em branco ou incoerente.\n"
        "2 - Demonstra algum conhecimento mas está incompleto ou parcialmente errado.\n"
        "3 - Parcialmente correto — acertou parte do conceito.\n"
        "4 - Correto — demonstra compreensão, mesmo que em outras palavras.\n\n"
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
    
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.choices[0].message.content.strip()
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
            card_in_english = is_english(question)
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
