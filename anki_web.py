"""
Anki Voice Assistant — Versão Web
Servidor Flask + WebSocket acessível de um celular na mesma rede.

Uso:  python3 anki_web.py
Acesse:  http://<IP-do-PC>:5000  no navegador do celular
"""

import eventlet
eventlet.monkey_patch()

import os
import re
import time
import asyncio
import subprocess
import logging
import base64

import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from anki.collection import Collection
from html2text import html2text

from config import ANKI_PATH
from anki_ai import (
    ensure_models_loaded,
    transcribe,
    evaluate_response,
    make_latex_speakable,
    strip_punctuation_for_tts,
    strip_images_from_text,
    tts_to_bytes,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig()

app = Flask(__name__)
app.config["SECRET_KEY"] = "anki-voice-assistant"
socketio = SocketIO(app, max_http_buffer_size=50 * 1024 * 1024)  # 50MB para áudio

# Coleção Anki e estado do card atual (por sessão única — uso pessoal)
collection = None
media_dir = None
current_card = None


def get_collection():
    global collection, media_dir
    if collection is None:
        collection = Collection(ANKI_PATH)
        media_dir = collection.media.dir()
    return collection


def convert_audio_to_wav(audio_bytes):
    """Converte áudio do navegador para WAV via ffmpeg, garantindo compatibilidade."""
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        log.error(f"ffmpeg error: {proc.stderr.decode()}")
        return None
    return proc.stdout


def send_tts(text, event="tts_audio"):
    """Gera TTS e envia áudio MP3 via WebSocket."""
    import queue
    import threading
    q = queue.Queue()
    
    def _worker():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(tts_to_bytes(text))
            q.put(("ok", res))
        except Exception as e:
            q.put(("error", e))
        finally:
            loop.close()
            
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    
    status, result = q.get()
    if status == "error":
        raise result
        
    emit(event, {"audio": base64.b64encode(result).decode()})


def advance_card():
    """Avança para o próximo card válido. Retorna (card, question, answer_text) ou None."""
    global current_card
    col = get_collection()

    while True:
        card = col.sched.getCard()
        if card is None:
            current_card = None
            return None

        # Extrai pergunta e resposta já renderizadas (funciona para Cloze e qualquer formato customizado)
        try:
            question_html = card.question()
            answer_html = card.answer()
        except Exception as e:
            log.error(f"Erro ao renderizar card {card.id}: {e}")
            col.sched.bury_cards([card.id])
            continue

        answer_text = html2text(strip_images_from_text(answer_html)).strip()
        question = html2text(strip_images_from_text(question_html)).strip()

        if not answer_text:
            log.debug("Card sem texto de resposta. Passando para o usuário mesmo assim.")

        current_card = card
        return card, question, answer_text


@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/media/<path:filename>")
def serve_media(filename):
    """Serve arquivos de mídia do Anki (imagens dos cards)."""
    return send_from_directory(get_collection().media.dir(), filename)


def parse_deck_tree(node):
    decks = []
    if node.name != "":
        decks.append({
            "id": node.deck_id,
            "name": node.name,
            "new": getattr(node, 'new_count', 0),
            "learn": getattr(node, 'learn_count', 0),
            "review": getattr(node, 'review_count', 0),
            "level": getattr(node, 'level', 0)
        })
    for child in getattr(node, 'children', []):
        decks.extend(parse_deck_tree(child))
    return decks


@app.route("/decks")
def get_decks_api():
    col = get_collection()
    tree = col.sched.deck_due_tree()
    decks = parse_deck_tree(tree)
    return {"decks": decks}


@app.route("/sync", methods=["POST"])
def sync_api():
    user = os.getenv("ANKIWEB_USER")
    pw = os.getenv("ANKIWEB_PASSWORD")
    if not user or not pw:
        return {"success": False, "message": "Credenciais não configuradas no .env"}, 400
    
    col = get_collection()
    try:
        auth = col.sync_login(user, pw, None)
        col.sync_collection(auth, sync_media=True)
        return {"success": True, "message": "Sincronização concluída!"}
    except Exception as e:
        log.error(f"Erro na sincronização: {e}")
        return {"success": False, "message": str(e)}, 500


@socketio.on("connect")
def handle_connect():
    log.info("Cliente conectado")
    ensure_models_loaded()


@socketio.on("start_session")
def handle_start_session(data):
    """Inicia sessão de estudo focada no deck selecionado."""
    deck_id = data.get("deck_id")
    if deck_id:
        col = get_collection()
        col.decks.set_current(deck_id)
    send_next_card()


def send_next_card():
    """Envia o próximo card para o cliente."""
    result = advance_card()
    if result is None:
        emit("session_end", {"message": "Parabéns! Você terminou todos os cards de hoje!"})
        return

    card, question, answer_text = result
    card.timer_started = time.time()

    # HTML do card com URLs de imagens apontando para /media/
    card_html = card.render_output(browser=True).question_and_style()
    # Reescreve referências de mídia para usar a rota /media/
    card_html = re.sub(r'src="(?!http)', 'src="/media/', card_html)
    # Corrige duplicação se já tiver caminho
    card_html = card_html.replace('src="/media//media/', 'src="/media/')

    emit("card_html", {
        "html": card_html,
        "question_text": question,
        "answer_text": answer_text,
    })

    # Envia TTS da pergunta
    speakable = make_latex_speakable(question)
    send_tts(speakable, "question_tts")


@socketio.on("audio_answer")
def handle_audio_answer(data):
    """Recebe áudio da resposta do usuário, transcreve e avalia."""
    global current_card

    if current_card is None:
        emit("error", {"message": "Nenhum card ativo"})
        return

    card = current_card
    answer_text = data.get("answer_text", "")

    # Decodifica áudio base64 recebido do navegador
    audio_b64 = data.get("audio", "")
    audio_webm = base64.b64decode(audio_b64)

    emit("status", {"message": "Transcrevendo..."})
    eventlet.sleep(0)

    # Converter para WAV via ffmpeg para evitar problemas de codec (ex: Safari/iOS)
    audio_wav = convert_audio_to_wav(audio_webm)
    if not audio_wav:
        emit("error", {"message": "Erro interno ao processar áudio"})
        return

    import io
    wav_io = io.BytesIO(audio_wav)
    wav_io.name = "audio.wav"

    # Transcreve usando API (agora com formato WAV padronizado)
    user_response = transcribe(wav_io)
    log.debug(f"Transcrição: {user_response}")
    emit("transcription", {"text": user_response})
    eventlet.sleep(0)

    # Verifica comandos de voz
    if "skip card" in user_response.lower():
        col = get_collection()
        col.sched.bury_cards([card.id])
        send_next_card()
        return

    # Avalia
    emit("status", {"message": "Avaliando..."})
    eventlet.sleep(0)
    (question, _) = card.note().fields

    if "nao sei" in user_response.lower() or "não sei" in user_response.lower():
        score, feedback = 1, ""
    else:
        score, feedback = evaluate_response(question, answer_text, user_response)

    log.info(f"Score: {score} | Feedback: {feedback!r}")

    col = get_collection()
    col.sched.answerCard(card, score)

    # Envia resultado
    color = "#02CC0255" if score >= 4 else "#CC020255"
    emit("result", {
        "score": score,
        "feedback": feedback,
        "flash_color": color,
    })

    # Gera e envia TTS do feedback
    if score == 4:
        elogio = "Muito bom, acertou!"
        tts_text = f"{elogio} {feedback}" if feedback else elogio
    else:
        answer_spoken = strip_punctuation_for_tts(answer_text)
        tts_text = f"Você errou. {feedback} A resposta correta é: {answer_spoken}" if feedback else f"Você errou. A resposta correta é: {answer_spoken}"

        # Envia HTML da resposta
        answer_html = card.render_output(browser=True).answer_and_style()
        answer_html = re.sub(r'src="(?!http)', 'src="/media/', answer_html)
        answer_html = answer_html.replace('src="/media//media/', 'src="/media/')
        emit("show_answer", {"html": answer_html})

    send_tts(tts_text, "feedback_tts")


@socketio.on("skip_card")
def handle_skip_card():
    """Enterra o card atual e avança."""
    global current_card
    if current_card:
        col = get_collection()
        col.sched.bury_cards([current_card.id])
    send_next_card()


@socketio.on("next_card")
def handle_next_card():
    """Avança para o próximo card."""
    send_next_card()


@socketio.on("disconnect")
def handle_disconnect():
    log.info("Cliente desconectado")


if __name__ == "__main__":
    import socket

    ensure_models_loaded()

    # Mostra o IP local para fácil acesso do celular
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    # Tenta pegar o IP real da rede (não 127.0.0.1)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print(f"\n{'='*50}")
    print(f"  Anki Voice Assistant — Versão Web")
    print(f"  Servidor iniciado na porta 5001.")
    print(f"{'='*50}\n")

    socketio.run(app, host="0.0.0.0", port=5001)
