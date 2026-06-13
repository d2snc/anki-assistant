"""
Anki Voice Assistant — Versão Web
Servidor Flask + WebSocket acessível de um celular na mesma rede.

Uso:  python3 anki_web.py
Acesse:  http://<IP-do-PC>:5000  no navegador do celular
"""

import eventlet
eventlet.monkey_patch()
from eventlet import tpool

import os
import re
import json
import time
import asyncio
import subprocess
import logging
import base64
import unicodedata
from datetime import datetime

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
    PASS_SCORE,
    is_english,
    TTS_VOICE_EN,
    make_latex_speakable,
    strip_punctuation_for_tts,
    strip_images_from_text,
    get_prioritized_card,
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
current_card_english = False  # True quando o card atual está em inglês
current_question = None  # texto renderizado da pergunta do card atual (para avaliação)
session_stats = None  # métricas da sessão de estudo em andamento (None = sem sessão)

# Histórico de sessões: uma lista JSON em disco, append-only. Fica fora do git
# (uso pessoal) e mora ao lado deste arquivo para não depender do CWD.
SESSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_history.json")

# Tag aplicada às notas marcadas como "cartão ruim". A galeria é derivada ao vivo
# desta tag (sem arquivo paralelo): suspende-se o card para não reaparecer no
# estudo até o usuário editá-lo no Anki e removê-lo da galeria (untag + unsuspend).
BAD_CARD_TAG = "cartao-ruim"


def get_collection():
    global collection, media_dir
    if collection is None:
        collection = Collection(ANKI_PATH)
        media_dir = collection.media.dir()
    return collection


# ---------------------------------------------------------------------------
# Histórico e estatísticas por sessão de estudo
# ---------------------------------------------------------------------------

def start_session_stats(deck_name):
    """Inicia o acumulador de métricas de uma nova sessão."""
    global session_stats
    session_stats = {
        "deck": deck_name,
        "started_at": time.time(),
        "answered": 0,   # cards efetivamente respondidos (acerto + erro)
        "correct": 0,
        "wrong": 0,
        "skipped": 0,    # cards pulados pelo usuário (botão/voz), não os auto-enterrados
        "scores": [],    # notas 1–4 de cada card respondido
    }


def record_answer(passed, score):
    """Contabiliza um card respondido na sessão atual."""
    if session_stats is None:
        return
    session_stats["answered"] += 1
    session_stats["scores"].append(score)
    if passed:
        session_stats["correct"] += 1
    else:
        session_stats["wrong"] += 1


def record_skip():
    """Contabiliza um card pulado pelo usuário na sessão atual."""
    if session_stats is not None:
        session_stats["skipped"] += 1


def load_history():
    """Lê o histórico de sessões do disco. Lista vazia se ainda não existe."""
    try:
        with open(SESSIONS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def finalize_session():
    """Fecha a sessão atual, grava no histórico e devolve o resumo (ou None).

    Sessões sem nenhuma atividade (nada respondido nem pulado) não são gravadas.
    Idempotente: zera session_stats, então chamadas repetidas (ex.: session_end
    seguido de disconnect) não duplicam o registro.
    """
    global session_stats
    if session_stats is None:
        return None

    stats = session_stats
    session_stats = None

    if stats["answered"] == 0 and stats["skipped"] == 0:
        return None

    ended = time.time()
    answered = stats["answered"]
    accuracy = round(100 * stats["correct"] / answered, 1) if answered else 0.0
    avg_score = round(sum(stats["scores"]) / answered, 2) if answered else 0.0

    record = {
        "deck": stats["deck"],
        "started_at": datetime.fromtimestamp(stats["started_at"]).isoformat(timespec="seconds"),
        "ended_at": datetime.fromtimestamp(ended).isoformat(timespec="seconds"),
        "duration_sec": round(ended - stats["started_at"], 1),
        "answered": answered,
        "correct": stats["correct"],
        "wrong": stats["wrong"],
        "skipped": stats["skipped"],
        "accuracy": accuracy,
        "avg_score": avg_score,
    }

    history = load_history()
    history.append(record)
    try:
        with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        log.error(f"Erro ao gravar histórico de sessão: {e}")

    log.info(f"Sessão registrada: {record}")
    return record


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


def send_tts(text, event="tts_audio", voice=None):
    """Gera TTS e envia áudio MP3 via WebSocket.

    O edge-tts roda em asyncio. Misturar um loop asyncio com o hub do eventlet
    (que monkey-patcha o threading) provoca "Cannot run the event loop while
    another loop is running" quando duas chamadas se intercalam. Por isso a
    coroutine é executada via eventlet.tpool, que roda num thread de SO real e
    isolado do hub, enquanto cede o controle para atender outros clientes.
    """
    audio = tpool.execute(lambda: asyncio.run(tts_to_bytes(text, voice=voice)))
    emit(event, {"audio": base64.b64encode(audio).decode()})


def is_basic_note(card):
    """True se o card é de um note type Basic/Básico (frente/verso simples).

    Normaliza acentos para que 'Básico' (Anki em português) também conte, e
    cobre as variantes oficiais ('Basic (and reversed card)' etc.). Cloze e
    outros note types são ignorados pela sessão de voz.
    """
    name = card.note_type()["name"]
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c)).lower()
    return "basic" in name


# Filas do Anki (campo queue do card): 0=new, 1=learning intradiário, 2=review
# (Due), 3=day-learn. Só a 1 (learning intradiário) trava no topo da fila.
QUEUE_LEARN_INTRADAY = 1
# Quantos cards à frente vasculhar para pular a "frente" de learning e achar o Due.
# Generoso de propósito: cobre o caso de muitos learning empilhados antes do Due.
CARD_LOOKAHEAD = 500


def _pick_next_card(col):
    """Escolhe o próximo card priorizando Due/New sobre o learning intradiário.

    A regra do backend v3 (descoberta empiricamente): você só pode responder o
    PRIMEIRO card da fila que NÃO é learning intradiário (queue 1). Os learning da
    frente podem ser "pulados" — então revisões e cards novos saem antes de a gente
    gastar tempo no learning, que é o que fazia o Due nunca fechar. Mas NÃO dá para
    reordenar entre os não-learning: responder um review/novo mais fundo (pulando um
    da frente, ex.: um não-Basic) devolve 'not at top of queue'.

    Por isso o predicado filtra só por queue != 1 (sem olhar Basic): devolvemos o
    verdadeiro primeiro não-learning. Se ele for não-Basic ou não renderizar, quem
    enterra é advance_card — nunca pulamos para um mais fundo. Sem nenhum
    não-learning, caímos no topo real (learning), que é sempre respondível.
    """
    card = get_prioritized_card(
        col, fetch_limit=CARD_LOOKAHEAD,
        predicate=lambda c: getattr(c, "queue", None) != QUEUE_LEARN_INTRADAY,
    )
    if card is not None:
        return card

    # Só sobrou learning intradiário: responde o topo real da fila.
    return get_prioritized_card(col)


def advance_card():
    """Avança para o próximo card estudável. Retorna (card, question, answer_text) ou None.

    A escolha (prioridade Due → New → Learn) fica em _pick_next_card. Aqui só
    tratamos o que ele devolve: cards fora do escopo da sessão de voz (não-Basic)
    ou que não renderizam são enterrados (bury) para a fila avançar — bury não
    altera intervalo/ease e o Anki desenterra sozinho no dia seguinte. Resposta
    ERRADA nunca passa por aqui — ela vai pra learn via Again no fluxo natural.
    """
    global current_card
    col = get_collection()

    while True:
        card = _pick_next_card(col)
        if card is None:
            current_card = None
            return None

        # _pick_next_card devolve o 1º não-learning (Basic ou não). Se for não-Basic,
        # enterra para destravar — nunca pulamos para um Basic mais fundo (daria
        # 'not at top of queue'). É o único ponto que altera a fila aqui.
        if not is_basic_note(card):
            col.sched.bury_cards([card.id])
            continue

        # Extrai pergunta e resposta já renderizadas (funciona para qualquer formato customizado)
        try:
            question_html = card.question()
            answer_html = card.answer()
        except Exception as e:
            log.error(f"Erro ao renderizar card {card.id}: {e}")
            col.sched.bury_cards([card.id])  # remove o card quebrado do topo da fila
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


@app.route("/history")
def get_history_api():
    """Histórico de sessões, mais recentes primeiro, com totais agregados."""
    sessions = load_history()
    totals = {
        "sessions": len(sessions),
        "answered": sum(s.get("answered", 0) for s in sessions),
        "correct": sum(s.get("correct", 0) for s in sessions),
        "duration_sec": round(sum(s.get("duration_sec", 0) for s in sessions), 1),
    }
    answered = totals["answered"]
    totals["accuracy"] = round(100 * totals["correct"] / answered, 1) if answered else 0.0
    return {"sessions": list(reversed(sessions)), "totals": totals}


@app.route("/bad_cards")
def get_bad_cards_api():
    """Galeria de cartões ruins: notas com a tag BAD_CARD_TAG, renderizadas ao
    vivo (pergunta/resposta/deck) para o usuário localizar e editar no Anki."""
    col = get_collection()
    cards = []
    for nid in col.find_notes(f"tag:{BAD_CARD_TAG}"):
        note = col.get_note(nid)
        note_cards = note.cards()
        if not note_cards:
            continue
        card = note_cards[0]
        try:
            question = html2text(strip_images_from_text(card.question())).strip()
            answer = html2text(strip_images_from_text(card.answer())).strip()
        except Exception as e:
            log.error(f"Erro ao renderizar card ruim {card.id}: {e}")
            question, answer = "", ""
        deck_id = getattr(card, "odid", 0) or card.did
        cards.append({
            "note_id": nid,
            "deck": col.decks.name(deck_id),
            "question": question,
            "answer": answer,
        })
    return {"cards": cards}


@app.route("/bad_cards/<int:note_id>", methods=["DELETE"])
def remove_bad_card_api(note_id):
    """Remove uma nota da galeria: tira a tag e reativa (unsuspend) os cards,
    devolvendo-os ao estudo. Chamado depois que o usuário editou no Anki."""
    col = get_collection()
    try:
        note = col.get_note(note_id)
    except Exception:
        return {"success": False, "message": "Nota não encontrada"}, 404
    card_ids = [c.id for c in note.cards()]
    col.tags.bulk_remove([note_id], BAD_CARD_TAG)
    if card_ids:
        col.sched.unsuspend_cards(card_ids)
    return {"success": True}


@socketio.on("connect")
def handle_connect():
    log.info("Cliente conectado")
    ensure_models_loaded()


@socketio.on("start_session")
def handle_start_session(data):
    """Inicia sessão de estudo focada no deck selecionado."""
    finalize_session()  # grava sessão anterior não finalizada, se houver
    deck_id = data.get("deck_id")
    col = get_collection()
    deck_name = "Todos os decks"
    if deck_id:
        col.decks.set_current(deck_id)
        deck_name = col.decks.name(deck_id)
    start_session_stats(deck_name)
    send_next_card()


# Mapeia a fila (queue)/tipo (type) do Anki para a categoria exibida na tela.
# queue/type: 0=new, 1=learning, 2=review, 3=day-learn/relearn.
_CARD_STATE = {0: "new", 1: "learn", 2: "review", 3: "learn"}


def card_state(card):
    """Retorna 'new', 'learn' ou 'review' conforme a fila do card atual.

    Usa a fila (queue) como primário e o tipo (type) como fallback, já que
    ambos compartilham a mesma codificação para as filas ativas.
    """
    state = _CARD_STATE.get(getattr(card, "queue", None))
    if state is None:
        state = _CARD_STATE.get(getattr(card, "type", 2), "review")
    return state


def emit_stats(card=None):
    """Envia os contadores do deck (new/learn/review) e, quando há um card
    ativo, em qual baralho e fila (new/learn/due) ele está."""
    try:
        new, learn, review = get_collection().sched.counts()
    except Exception as e:
        log.error(f"Erro ao obter contadores: {e}")
        return
    payload = {"new": new, "learn": learn, "review": review}
    if card is not None:
        # odid != 0 quando o card está num deck filtrado; nesse caso o deck
        # "de verdade" do card é o original (odid).
        deck_id = getattr(card, "odid", 0) or card.did
        payload["current_deck"] = get_collection().decks.name(deck_id)
        payload["current_state"] = card_state(card)
    emit("stats", payload)


def send_next_card():
    """Envia o próximo card para o cliente."""
    global current_card_english, current_question
    result = advance_card()
    if result is None:
        summary = finalize_session()
        payload = {"message": "Parabéns! Você terminou todos os cards de hoje!"}
        if summary:
            payload["summary"] = summary
        emit("session_end", payload)
        return

    card, question, answer_text = result
    card.timer_started = time.time()
    current_question = question

    current_card_english = is_english(question)
    voice = TTS_VOICE_EN if current_card_english else None

    # HTML do card com URLs de imagens apontando para /media/
    card_html = card.render_output(browser=True).question_and_style()
    # Reescreve referências de mídia para usar a rota /media/
    card_html = re.sub(r'src="(?!http)', 'src="/media/', card_html)
    # Corrige duplicação se já tiver caminho
    card_html = card_html.replace('src="/media//media/', 'src="/media/')

    emit_stats(card)
    emit("card_html", {
        "html": card_html,
        "question_text": question,
        "answer_text": answer_text,
    })

    # Envia TTS da pergunta
    speakable = make_latex_speakable(question)
    speakable = strip_punctuation_for_tts(speakable)
    if not speakable:
        speakable = "Verifique a tela."
    send_tts(speakable, "question_tts", voice=voice)


@socketio.on("audio_answer")
def handle_audio_answer(data):
    """Recebe áudio da resposta do usuário, transcreve e avalia."""
    global current_card, current_card_english

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
    user_response = transcribe(wav_io, lang="en" if current_card_english else "pt")
    log.debug(f"Transcrição: {user_response}")
    emit("transcription", {"text": user_response})
    eventlet.sleep(0)

    # Verifica comandos de voz
    if "skip card" in user_response.lower():
        get_collection().sched.bury_cards([card.id])
        record_skip()
        send_next_card()
        return

    # Avalia
    emit("status", {"message": "Avaliando..."})
    eventlet.sleep(0)
    # Usa a pergunta já renderizada (funciona para qualquer note type, inclusive
    # Cloze ou com 3+ campos, que quebravam ao desempacotar note().fields em 2).
    question = current_question if current_question is not None else card.note().fields[0]

    lang = "en" if current_card_english else "pt"
    if "nao sei" in user_response.lower() or "não sei" in user_response.lower():
        score, feedback = 1, ""
    else:
        score, feedback = evaluate_response(question, answer_text, user_response, lang=lang)

    log.info(f"Score: {score} | Feedback: {feedback!r}")

    col = get_collection()
    # Desacopla o veredito (acertou/errou) do botão do Anki (ease): acerto
    # (score >= PASS_SCORE) mantém o ease do avaliador (Good/Easy) e respeita o
    # intervalo; erro vira Again (1) para o Anki reaprender o card.
    passed = score >= PASS_SCORE
    ease = score if passed else 1
    try:
        col.sched.answerCard(card, ease)
    except Exception as e:
        log.error(f"Erro ao responder card: {e}")
        send_next_card()
        return
    # Card consumido: zera o estado para que um audio_answer duplicado/atrasado
    # (race de áudio ou rede) não tente respondê-lo de novo. O próximo card é
    # definido por send_next_card (disparado pelo cliente após o feedback_tts).
    current_card = None
    record_answer(passed, score)

    # Envia resultado
    color = "#02CC0255" if passed else "#CC020255"
    emit("result", {
        "score": score,
        "feedback": feedback,
        "flash_color": color,
    })

    answer_spoken = strip_punctuation_for_tts(answer_text)
    if passed:
        if current_card_english:
            elogio = f"Well done! The answer is: {answer_spoken}" if answer_spoken else "Well done!"
        else:
            elogio = f"Muito bom, acertou! A resposta é: {answer_spoken}" if answer_spoken else "Muito bom, acertou!"
        tts_text = f"{elogio} {feedback}" if feedback else elogio
    else:
        if current_card_english:
            tts_text = f"Wrong. {feedback} The correct answer is: {answer_spoken}" if feedback else f"Wrong. The correct answer is: {answer_spoken}"
        else:
            tts_text = f"Você errou. {feedback} A resposta correta é: {answer_spoken}" if feedback else f"Você errou. A resposta correta é: {answer_spoken}"

        # Envia HTML da resposta
        answer_html = card.render_output(browser=True).answer_and_style()
        answer_html = re.sub(r'src="(?!http)', 'src="/media/', answer_html)
        answer_html = answer_html.replace('src="/media//media/', 'src="/media/')
        emit("show_answer", {"html": answer_html})

    # Atualiza o painel imediatamente após responder: o card.load() feito por
    # answerCard já deixou o card no estado novo (ex.: erro → fila 'learn'),
    # então new/learn/due refletem a resposta sem esperar o próximo card.
    emit_stats(card)

    voice = TTS_VOICE_EN if current_card_english else None
    send_tts(tts_text, "feedback_tts", voice=voice)


@socketio.on("skip_card")
def handle_skip_card():
    """Pula o card atual enterrando-o (bury). Apenas "esconder na sessão" não
    funciona: o card continuaria no topo da fila do Anki e bloquearia
    ('not at top of queue') os cards atrás dele. Bury tira o card da frente da
    fila sem mudar intervalo/ease, e o Anki o desenterra no dia seguinte."""
    if current_card:
        get_collection().sched.bury_cards([current_card.id])
        record_skip()
    send_next_card()


@socketio.on("flag_bad_card")
def handle_flag_bad_card():
    """Marca o card atual como 'cartão ruim': aplica a tag na nota e suspende
    todos os cards dela, de modo que não reapareça no estudo. A nota passa a
    aparecer na galeria, onde o usuário a edita no Anki e depois a remove
    (untag + unsuspend). Em seguida avança para o próximo card."""
    global current_card
    if current_card:
        col = get_collection()
        note = current_card.note()
        card_ids = [c.id for c in note.cards()]
        col.tags.bulk_add([note.id], BAD_CARD_TAG)
        col.sched.suspend_cards(card_ids)
        current_card = None
    send_next_card()


@socketio.on("next_card")
def handle_next_card():
    """Avança para o próximo card."""
    send_next_card()


@socketio.on("disconnect")
def handle_disconnect():
    log.info("Cliente desconectado")
    # Salva a sessão mesmo se o usuário fechou a aba sem terminar os cards.
    finalize_session()


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
