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
import hashlib
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
from anki.cards import Card
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
    review_flashcard,
    generate_illustration,
    generate_illustration_video,
    generate_mcq,
    tts_to_bytes,
    SAMPLE_RATE,
)
from html import escape

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
# Modo de resposta da sessão: "voice" (fala + avaliação por IA, só cards Basic) ou
# "buttons" (auto-avaliação Again/Hard/Good/Easy como no Anki, qualquer note type).
session_mode = "voice"
# Fluxo da sessão no modo opções: "order" (os cards vêm na ordem Due→New→Learn,
# como no modo voz) ou "list" (o usuário vê a lista de uma coluna — Learn/New/Due
# — e escolhe qual card responder fora de ordem, para reduzir a pilha).
session_flow = "order"
# Coluna (new/learn/review) exibida no momento no modo listagem — usada para
# reenviar a lista atualizada depois de responder um card.
session_list_column = "review"

# Ilustração gerada (modo opções) aguardando aprovação. Guardada entre o preview
# e o "Usar imagem" para não regenerar ao aprovar. {"note_id": int, "png": bytes}.
pending_illustration = None

# Vídeo ilustrativo gerado (modo opções) aguardando aprovação. Guardado entre o
# preview e o "Usar vídeo" para não regenerar ao aprovar. {"note_id": int,
# "mp4": bytes}.
pending_illustration_video = None

# Note type de múltipla escolha (já existente na coleção do usuário) usado para
# transformar um card difícil em questão de alternativas. Campos: question,
# optionA..optionZ, answer (a letra do gabarito), note e noteA..noteZ.
MCQ_NOTE_TYPE = "IKKZ__MCQ_26.PT_BR.NATIVE"
# Múltipla escolha gerada (modo opções) aguardando confirmação. Guardada entre o
# preview e o "Criar e substituir" (a criação apaga a nota original, então o
# usuário confere antes). {"note_id": int, "mcq": dict de generate_mcq}.
pending_mcq = None

# Histórico de sessões: uma lista JSON em disco, append-only. Fica fora do git
# (uso pessoal) e mora ao lado deste arquivo para não depender do CWD.
SESSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_history.json")

# Tag aplicada às notas marcadas como "cartão ruim". A galeria é derivada ao vivo
# desta tag (sem arquivo paralelo): suspende-se o card para não reaparecer no
# estudo até o usuário editá-lo no Anki e removê-lo da galeria (untag + unsuspend).
BAD_CARD_TAG = "cartao-ruim"

# Revisor de flashcards em background: percorre as notas Basic (só texto) e
# guarda sugestões de melhoria/divisão num cache em disco (fora do git).
REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_suggestions.json")
REVIEW_DELAY_SEC = 3        # pausa entre chamadas de LLM (respeita rate limit do free tier)
REVIEW_RESCAN_SEC = 600     # após varrer tudo, espera antes de reprocurar notas novas/editadas

review_state = None         # {"analyzed": {nid: hash}, "suggestions": {nid: {...}}}
review_worker_started = False
review_progress = {"analyzed": 0, "total": 0, "running": False}


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


# O template Back do Basic é `{{FrontSide}}<hr id=answer>{{Back}}`, então a
# resposta renderizada repete a pergunta antes do <hr id=answer>. Casa o hr com
# ou sem aspas no id (variações do Anki).
_ANSWER_HR = re.compile(r'<hr\s+id=["\']?answer["\']?[^>]*>', re.IGNORECASE)


def answer_only_html(answer_html):
    """Devolve só a parte da resposta (Back), sem repetir a pergunta (FrontSide).

    Usado para o TTS não falar a pergunta de novo ao ler a resposta e para a
    avaliação comparar contra o gabarito limpo. Se não houver o <hr id=answer>
    (note types customizados), devolve o HTML inteiro como fallback.
    """
    parts = _ANSWER_HR.split(answer_html, maxsplit=1)
    return parts[1] if len(parts) > 1 else answer_html


def _media_rewritten(html):
    """Reescreve referências de mídia (src="arquivo") para a rota /media/ do
    servidor, sem duplicar o prefixo caso o caminho já o tenha."""
    html = re.sub(r'src="(?!http)', 'src="/media/', html)
    return html.replace('src="/media//media/', 'src="/media/')


# Filas do Anki (campo queue do card): 0=new, 1=learning intradiário, 2=review
# (Due), 3=day-learn. Só a 1 (learning intradiário) trava no topo da fila.
QUEUE_LEARN_INTRADAY = 1
# Quantos cards à frente vasculhar para pular a "frente" de learning e achar o Due.
# Generoso de propósito: cobre o caso de muitos learning empilhados antes do Due.
CARD_LOOKAHEAD = 500
# Tamanho da janela ao varrer a fila inteira (listagem + resposta fora de ordem).
# Tem que ser o MESMO nos dois lados: a lista é montada com este limite, então o
# card que o usuário escolhe precisa caber na mesma janela ao ser respondido —
# senão um card fundo (pilha grande) "não some" da lista ao ser respondido.
QUEUE_FETCH_LIMIT = 9999


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


def advance_card(skip_non_basic=True):
    """Avança para o próximo card estudável. Retorna (card, question, answer_text) ou None.

    A escolha (prioridade Due → New → Learn) fica em _pick_next_card. Aqui só
    tratamos o que ele devolve: cards que não renderizam são enterrados (bury)
    para a fila avançar — bury não altera intervalo/ease e o Anki desenterra
    sozinho no dia seguinte. Resposta ERRADA nunca passa por aqui — ela vai pra
    learn via Again no fluxo natural.

    skip_non_basic=True (modo voz): cards não-Basic são enterrados, pois a
    avaliação por IA precisa de pergunta/resposta em texto. No modo opções
    (skip_non_basic=False) o usuário se auto-avalia, então qualquer note type
    (Cloze etc.) é exibido — como no próprio Anki.
    """
    global current_card
    col = get_collection()

    while True:
        card = _pick_next_card(col)
        if card is None:
            current_card = None
            return None

        # _pick_next_card devolve o 1º não-learning (Basic ou não). No modo voz,
        # se for não-Basic, enterra para destravar — nunca pulamos para um Basic
        # mais fundo (daria 'not at top of queue'). É o único ponto que altera a
        # fila aqui (além do bury de card quebrado).
        if skip_non_basic and not is_basic_note(card):
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

        # Resposta SEM a pergunta repetida (FrontSide): vale tanto para o TTS
        # quanto para o gabarito da avaliação.
        answer_text = html2text(strip_images_from_text(answer_only_html(answer_html))).strip()
        question = html2text(strip_images_from_text(question_html)).strip()

        if not answer_text:
            log.debug("Card sem texto de resposta. Passando para o usuário mesmo assim.")

        current_card = card
        return card, question, answer_text


def next_button_labels(col, card):
    """Rótulos de intervalo dos 4 botões (Again/Hard/Good/Easy) tal como o Anki
    mostra (ex.: '1 min', '10 min', '4 d'), respeitando a configuração do deck.
    describe_next_states devolve sempre 4 strings na ordem again/hard/good/easy."""
    states = col._backend.get_scheduling_states(card.id)
    return list(col.sched.describe_next_states(states))


def _card_preview(card):
    """Texto curto da pergunta do card para a listagem (sem imagens/HTML)."""
    try:
        q = html2text(strip_images_from_text(card.question())).strip()
    except Exception as e:
        log.warning(f"Preview do card {card.id} falhou: {e}")
        q = ""
    q = " ".join(q.split())  # colapsa quebras/espaços
    if not q:
        return "(sem texto)"
    return q[:120] + "…" if len(q) > 120 else q


def collect_queue_by_state(col, fetch_limit=QUEUE_FETCH_LIMIT):
    """Agrupa a fila ativa do deck atual (na ordem do scheduler) por coluna
    new/learn/review, com um preview de texto de cada card. Reflete exatamente o
    que está estudável hoje — a mesma base dos contadores de counts()."""
    groups = {"new": [], "learn": [], "review": []}
    for queued in col.sched.get_queued_cards(fetch_limit=fetch_limit).cards:
        card = Card(col)
        card._load_from_backend_card(queued.card)
        groups.setdefault(card_state(card), []).append(
            {"id": card.id, "preview": _card_preview(card)}
        )
    return groups


def answer_specific_card(col, target_id, ease):
    """Responde um card ESPECÍFICO fora de ordem (modo listagem), contornando a
    regra do scheduler v3 de que só dá pra responder o topo da fila.

    Para isso enterra temporariamente os cards que ficam à frente do alvo na fila
    e os desenterra logo depois (no finally) — só os que ESTE método enterrou,
    pelos ids exatos, então skips/buries anteriores do usuário não são tocados.
    Cards de learning são preservados pelo par bury/unbury; e como learning pode
    ser "pulado" ao responder um não-learning, para alvo não-learning nem precisa
    enterrá-los. Para alvo learning (só o topo absoluto é respondível) enterra
    tudo à frente. Devolve o card respondido ou None se não der.

    O bury/unbury de toda a frente é pesado, mas é a operação que o Anki garante:
    confirmado num clone da coleção que não dá 'not at top of queue' e que os
    demais cards voltam intactos."""
    buried = []
    try:
        for _ in range(500):  # teto de segurança (cada passo enterra a janela toda)
            # Mesma janela da listagem: o card escolhido pode estar fundo numa
            # pilha grande, e precisa caber aqui para ser encontrado/respondido.
            cards = col.sched.get_queued_cards(fetch_limit=QUEUE_FETCH_LIMIT).cards
            by_id = {c.card.id: c for c in cards}
            if target_id not in by_id:
                # Alvo ainda não visível: se a janela veio cheia, pode haver mais
                # cards atrás — enterra a frente não-learning para trazê-los à
                # tona e tenta de novo. Senão, o alvo realmente sumiu (já respondido).
                if len(cards) >= QUEUE_FETCH_LIMIT:
                    front = [
                        c.card.id for c in cards
                        if c.card.queue != QUEUE_LEARN_INTRADAY
                    ]
                    if front:
                        col.sched.bury_cards(front, manual=True)
                        buried.extend(front)
                        continue
                return None  # alvo saiu da fila (já respondido?) — nada a fazer
            target_is_learn = by_id[target_id].card.queue == QUEUE_LEARN_INTRADAY
            if target_is_learn:
                # Learning só é respondível no topo absoluto da fila.
                answerable = cards[0].card.id == target_id
                to_bury = [c.card.id for c in cards if c.card.id != target_id]
            else:
                # Não-learning é respondível quando é o 1º não-learning (learning
                # à frente é pulado pelo scheduler).
                first_non_learn = next(
                    (c.card.id for c in cards if c.card.queue != QUEUE_LEARN_INTRADAY), None
                )
                answerable = first_non_learn == target_id
                to_bury = [
                    c.card.id for c in cards
                    if c.card.id != target_id and c.card.queue != QUEUE_LEARN_INTRADAY
                ]
            if answerable:
                card = Card(col)
                card._load_from_backend_card(by_id[target_id].card)
                card.start_timer()
                col.sched.answerCard(card, ease)
                return card
            if not to_bury:
                return None  # nada a enterrar e ainda não é o topo: desiste
            col.sched.bury_cards(to_bury, manual=True)
            buried.extend(to_bury)
        return None
    finally:
        if buried:
            col.sched.unbury_cards(buried)


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


# ---------------------------------------------------------------------------
# Revisor de flashcards em background
# ---------------------------------------------------------------------------

def load_review_state():
    """Carrega (uma vez) o cache de sugestões do disco para a memória."""
    global review_state
    if review_state is not None:
        return review_state
    try:
        with open(REVIEW_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    review_state = {
        "analyzed": data.get("analyzed", {}),       # nid (str) -> hash do conteúdo já analisado
        "suggestions": data.get("suggestions", {}),  # nid (str) -> sugestão pendente
    }
    return review_state


def save_review_state():
    """Persiste o cache de sugestões no disco."""
    if review_state is None:
        return
    try:
        with open(REVIEW_PATH, "w", encoding="utf-8") as f:
            json.dump(review_state, f, ensure_ascii=False, indent=2)
    except OSError as e:
        log.error(f"Erro ao gravar sugestões de revisão: {e}")


def _is_basic_notetype(note):
    """True se a NOTA é de um note type Basic/Básico. Mesma normalização de
    acentos/variantes de is_basic_note, mas a partir da nota (não do card)."""
    name = note.note_type()["name"]
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c)).lower()
    return "basic" in name


def _has_media(html):
    """True se o campo contém imagem ou som — esses cartões são pulados pelo
    revisor para não destruir mídia ao reescrever os campos em texto."""
    return bool(re.search(r'<img|\[sound:|\[anki:play:', html, re.IGNORECASE))


def _note_text(html):
    """Converte o HTML de um campo para texto limpo (sem imagens)."""
    return html2text(strip_images_from_text(html)).strip()


def _content_hash(front_html, back_html):
    """Hash do conteúdo da nota: muda quando a frente/verso são editados, o que
    faz o revisor reanalisar a nota num passe futuro."""
    return hashlib.sha1((front_html + "\x00" + back_html).encode("utf-8")).hexdigest()


def _note_deck_name(col, note):
    """Nome do deck do primeiro card da nota (usa o deck original se filtrado)."""
    cards = note.cards()
    if not cards:
        return ""
    deck_id = getattr(cards[0], "odid", 0) or cards[0].did
    return col.decks.name(deck_id)


def _to_field_html(text):
    """Texto plano vindo do LLM → HTML simples de campo do Anki (\\n vira <br>)."""
    return text.replace("\n", "<br>")


def _collect_review_candidates(col):
    """Notas elegíveis para revisão: Basic, só texto, com frente e verso.
    Devolve lista de (nid, note, front_text, back_text, content_hash)."""
    candidates = []
    for nid in col.find_notes("deck:*"):
        try:
            note = col.get_note(nid)
        except Exception:
            continue
        if not _is_basic_notetype(note) or len(note.fields) < 2:
            continue
        front_html, back_html = note.fields[0], note.fields[1]
        if _has_media(front_html) or _has_media(back_html):
            continue
        front, back = _note_text(front_html), _note_text(back_html)
        if not front and not back:
            continue
        candidates.append((nid, note, front, back, _content_hash(front_html, back_html)))
    return candidates


def review_worker():
    """Worker cooperativo (eventlet): varre as notas Basic, pede ao LLM uma
    avaliação de cada uma ainda não analisada e guarda as sugestões. Entre as
    chamadas de LLM cede o controle (eventlet.sleep) para não travar a sessão de
    estudo. Após varrer tudo, dorme e reprocura notas novas/editadas."""
    log.info("Worker de revisão de flashcards iniciado.")
    state = load_review_state()
    while True:
        try:
            col = get_collection()
            candidates = _collect_review_candidates(col)
        except Exception as e:
            log.error(f"Revisor: erro ao listar notas: {e}")
            eventlet.sleep(60)
            continue

        review_progress["total"] = len(candidates)
        review_progress["analyzed"] = sum(
            1 for (nid, _, _, _, h) in candidates if state["analyzed"].get(str(nid)) == h
        )
        review_progress["running"] = True

        for (nid, note, front, back, content_hash) in candidates:
            key = str(nid)
            if state["analyzed"].get(key) == content_hash:
                continue  # já analisado e inalterado

            try:
                result = review_flashcard(front, back)
                if result is None:
                    # LLM indisponível agora; não marca como analisado para tentar
                    # de novo num próximo passe. Espera para não martelar o rate limit.
                    eventlet.sleep(REVIEW_DELAY_SEC)
                    continue

                state["analyzed"][key] = content_hash
                if result["verdict"] in ("edit", "split") and result["cards"]:
                    state["suggestions"][key] = {
                        "note_id": nid,
                        "deck": _note_deck_name(col, note),
                        "verdict": result["verdict"],
                        "reason": result["reason"],
                        "current": {"front": front, "back": back},
                        "cards": result["cards"],
                        "status": "pending",
                        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                else:
                    # Virou "ok" (ou mudou e agora está bom): limpa sugestão antiga.
                    state["suggestions"].pop(key, None)

                review_progress["analyzed"] += 1
                save_review_state()
            except Exception as e:
                log.error(f"Revisor: erro ao analisar nota {nid}: {e}")
            eventlet.sleep(REVIEW_DELAY_SEC)

        review_progress["running"] = False
        eventlet.sleep(REVIEW_RESCAN_SEC)


def start_review_worker():
    """Inicia o worker de revisão uma única vez."""
    global review_worker_started
    if review_worker_started:
        return
    review_worker_started = True
    socketio.start_background_task(review_worker)


@app.route("/review_suggestions")
def get_review_suggestions_api():
    """Sugestões de revisão pendentes + progresso do worker em background."""
    state = load_review_state()
    suggestions = [s for s in state["suggestions"].values() if s.get("status") == "pending"]
    suggestions.sort(key=lambda s: s.get("analyzed_at", ""), reverse=True)
    return {"suggestions": suggestions, "progress": review_progress}


def _apply_edit(col, note, card):
    """Aplica uma sugestão de edição: reescreve frente/verso da própria nota."""
    note.fields[0] = _to_field_html(card.get("front", ""))
    note.fields[1] = _to_field_html(card.get("back", ""))
    col.update_note(note)


def _apply_split(col, note, cards):
    """Aplica uma sugestão de divisão: cria os cartões menores (mesmo note type,
    deck e tags) e remove a nota grande original."""
    notetype = note.note_type()
    orig_cards = note.cards()
    if orig_cards:
        deck_id = getattr(orig_cards[0], "odid", 0) or orig_cards[0].did
    else:
        deck_id = col.decks.get_current_id()
    for card in cards:
        new = col.new_note(notetype)
        new.fields[0] = _to_field_html(card.get("front", ""))
        new.fields[1] = _to_field_html(card.get("back", ""))
        new.tags = list(note.tags)
        col.add_note(new, deck_id)
    col.remove_notes([note.id])


@app.route("/review_suggestions/<int:note_id>/accept", methods=["POST"])
def accept_review_suggestion_api(note_id):
    """Aceita e aplica uma sugestão: edita a nota ou a divide em cartões menores."""
    state = load_review_state()
    key = str(note_id)
    sugg = state["suggestions"].get(key)
    if not sugg or sugg.get("status") != "pending":
        return {"success": False, "message": "Sugestão não encontrada"}, 404

    col = get_collection()
    try:
        note = col.get_note(note_id)
    except Exception:
        # Nota apagada no Anki desde a análise: limpa a sugestão órfã.
        state["suggestions"].pop(key, None)
        state["analyzed"].pop(key, None)
        save_review_state()
        return {"success": False, "message": "Nota não existe mais"}, 404

    cards = sugg.get("cards", [])
    if not cards:
        return {"success": False, "message": "Sugestão sem conteúdo"}, 400

    try:
        if sugg["verdict"] == "edit":
            _apply_edit(col, note, cards[0])
        elif sugg["verdict"] == "split":
            _apply_split(col, note, cards)
        else:
            return {"success": False, "message": "Tipo de sugestão inválido"}, 400
    except Exception as e:
        log.error(f"Erro ao aplicar sugestão na nota {note_id}: {e}")
        return {"success": False, "message": str(e)}, 500

    # Some com a sugestão e esquece o hash antigo: o conteúdo mudou, então o
    # worker reanalisa a(s) nova(s) nota(s) num próximo passe.
    state["suggestions"].pop(key, None)
    state["analyzed"].pop(key, None)
    save_review_state()
    return {"success": True}


@app.route("/review_suggestions/<int:note_id>/dismiss", methods=["POST"])
def dismiss_review_suggestion_api(note_id):
    """Dispensa uma sugestão sem alterar o cartão. Mantém o hash em 'analyzed',
    então ela só reaparece se o conteúdo do cartão for editado no Anki."""
    state = load_review_state()
    key = str(note_id)
    if key in state["suggestions"]:
        state["suggestions"].pop(key, None)
        save_review_state()
    return {"success": True}


@socketio.on("connect")
def handle_connect():
    log.info("Cliente conectado")
    ensure_models_loaded()
    start_review_worker()


@socketio.on("start_session")
def handle_start_session(data):
    """Inicia sessão de estudo focada no deck selecionado.

    data['mode']: 'voice' (padrão — resposta falada avaliada por IA) ou 'buttons'
    (auto-avaliação Again/Hard/Good/Easy, como no Anki).
    data['flow'] (só no modo 'buttons'): 'order' (padrão — cards na ordem
    Due→New→Learn) ou 'list' (o usuário escolhe da lista qual card responder)."""
    global session_mode, session_flow
    finalize_session()  # grava sessão anterior não finalizada, se houver
    deck_id = data.get("deck_id")
    session_mode = "buttons" if data.get("mode") == "buttons" else "voice"
    session_flow = "list" if (session_mode == "buttons" and data.get("flow") == "list") else "order"
    col = get_collection()
    deck_name = "Todos os decks"
    if deck_id:
        col.decks.set_current(deck_id)
        deck_name = col.decks.name(deck_id)
    start_session_stats(deck_name)
    if session_flow == "list":
        emit_card_list()  # modo listagem: abre na lista, sem servir card ainda
    else:
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


def session_eta(new, learn, review):
    """Estimativa de tempo para fechar a fila ATIVA, seguindo a prioridade
    Due → New → Learn: enquanto houver Due, estima o tempo para zerar o Due;
    quando o Due acaba, passa a estimar o New; por fim, o Learn.

    O ritmo é o tempo real médio por flashcard da sessão (tempo decorrido /
    cards processados). Devolve {phase, remaining, eta_sec} ou None enquanto
    ainda não há ritmo medido (nenhum card processado) ou nada a fazer."""
    if session_stats is None:
        return None
    processed = session_stats["answered"] + session_stats["skipped"]
    if processed < 1:
        return None  # sem ritmo medido ainda
    avg = (time.time() - session_stats["started_at"]) / processed

    if review > 0:
        phase, remaining = "review", review
    elif new > 0:
        phase, remaining = "new", new
    elif learn > 0:
        phase, remaining = "learn", learn
    else:
        return None

    return {"phase": phase, "remaining": remaining, "eta_sec": round(avg * remaining)}


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
    eta = session_eta(new, learn, review)
    if eta is not None:
        payload["eta"] = eta
    emit("stats", payload)


def send_next_card():
    """Envia o próximo card para o cliente, conforme o modo da sessão.

    A prioridade Due → New → Learn é a mesma nos dois modos (vem de
    advance_card). Modo 'voice' (padrão): só cards Basic, com TTS da pergunta.
    Modo 'buttons': qualquer note type, sem TTS — o usuário se auto-avalia com
    Again/Hard/Good/Easy, como no Anki."""
    global current_question
    buttons_mode = session_mode == "buttons"
    result = advance_card(skip_non_basic=not buttons_mode)
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

    emit_stats(card)
    if buttons_mode:
        send_card_buttons(card)
    else:
        send_card_voice(card, question, answer_text)


def send_card_voice(card, question, answer_text):
    """Modo voz: envia a pergunta (HTML + texto) e o TTS para o cliente gravar a
    resposta, que será transcrita e avaliada pela IA."""
    global current_card_english
    current_card_english = is_english(question, answer_text)
    voice = TTS_VOICE_EN if current_card_english else None

    emit("card_html", {
        "html": _media_rewritten(card.render_output(browser=True).question_and_style()),
        "question_text": question,
        "answer_text": answer_text,
    })

    # Envia TTS da pergunta
    speakable = make_latex_speakable(question)
    speakable = strip_punctuation_for_tts(speakable)
    if not speakable:
        speakable = "Verifique a tela."
    send_tts(speakable, "question_tts", voice=voice)


def send_card_buttons(card):
    """Modo opções: envia pergunta e resposta já renderizadas e os rótulos de
    intervalo dos botões Again/Hard/Good/Easy. O cliente revela a resposta no
    'Mostrar resposta' e responde via evento answer_button. Sem TTS."""
    out = card.render_output(browser=True)
    emit("card_html", {
        "mode": "buttons",
        "html": _media_rewritten(out.question_and_style()),
        "answer_html": _media_rewritten(out.answer_and_style()),
        "buttons": next_button_labels(get_collection(), card),
    })


def emit_card_list(column=None):
    """Modo listagem: envia a lista de cards de cada coluna (Learn/New/Due) do
    deck atual mais os contadores, destacando a coluna `column` (mantém a
    anterior se None). O cliente mostra a lista e escolhe um card para responder."""
    global session_list_column
    if column in ("new", "learn", "review"):
        session_list_column = column
    col = get_collection()
    groups = collect_queue_by_state(col)
    emit("card_list", {
        "column": session_list_column,
        "cards": groups,
        "counts": {k: len(v) for k, v in groups.items()},
    })


@socketio.on("list_cards")
def handle_list_cards(data):
    """Modo listagem: o usuário trocou de coluna (ou pediu a lista). Reenvia."""
    emit_card_list(column=(data or {}).get("column"))


@socketio.on("pick_card")
def handle_pick_card(data):
    """Modo listagem: o usuário escolheu um card da lista para responder. Carrega
    e exibe esse card (pergunta + resposta + botões), como no modo opções. A
    resposta fora de ordem acontece no answer_button via answer_specific_card."""
    global current_card, current_question
    try:
        card_id = int((data or {}).get("card_id"))
    except (TypeError, ValueError):
        emit("error", {"message": "Card inválido"})
        return
    col = get_collection()
    try:
        card = col.get_card(card_id)
        question_html = card.question()
    except Exception as e:
        log.error(f"Erro ao abrir card {card_id} da lista: {e}")
        emit("error", {"message": "Não foi possível abrir esse card."})
        return
    current_card = card
    current_question = html2text(strip_images_from_text(question_html)).strip()
    emit_stats(card)
    send_card_buttons(card)


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

    # Mostra a resposta na tela (com imagem) tanto no acerto quanto no erro.
    answer_html = _media_rewritten(card.render_output(browser=True).answer_and_style())
    emit("show_answer", {"html": answer_html})

    # Atualiza o painel imediatamente após responder: o card.load() feito por
    # answerCard já deixou o card no estado novo (ex.: erro → fila 'learn'),
    # então new/learn/due refletem a resposta sem esperar o próximo card.
    emit_stats(card)

    voice = TTS_VOICE_EN if current_card_english else None
    send_tts(tts_text, "feedback_tts", voice=voice)


@socketio.on("answer_button")
def handle_answer_button(data):
    """Modo opções: o usuário se auto-avalia clicando Again/Hard/Good/Easy
    (ease 1–4), como no Anki. Responde o card com esse ease e avança. A
    prioridade Due → New → Learn é preservada — o próximo card sai de
    send_next_card → advance_card, igual ao modo voz."""
    global current_card
    if current_card is None:
        emit("error", {"message": "Nenhum card ativo"})
        return
    try:
        ease = int(data.get("ease", 0))
    except (TypeError, ValueError):
        ease = 0
    if ease not in (1, 2, 3, 4):
        emit("error", {"message": "Avaliação inválida"})
        return

    card = current_card
    col = get_collection()

    # Modo listagem: o card escolhido não está no topo da fila, então usa
    # answer_specific_card (enterra a frente, responde, desenterra). Depois volta
    # para a lista atualizada em vez de servir o próximo card automaticamente.
    if session_flow == "list":
        current_card = None
        answered = answer_specific_card(col, card.id, ease)
        if answered is None:
            emit("error", {"message": "Não foi possível responder esse card agora."})
            emit_card_list()
            return
        record_answer(passed=ease != 1, score=ease)
        emit_card_list()
        return

    try:
        # ease 1/2/3/4 mapeia exatamente para Again/Hard/Good/Easy no scheduler v3.
        col.sched.answerCard(card, ease)
    except Exception as e:
        log.error(f"Erro ao responder card (modo opções): {e}")
        current_card = None
        send_next_card()
        return

    # Card consumido: zera o estado antes de avançar (evita respostas duplicadas).
    current_card = None
    # Como no Anki, só "Again" (1) é lapso; Hard/Good/Easy contam como acerto.
    record_answer(passed=ease != 1, score=ease)
    emit_stats(card)
    send_next_card()


# Campos cujo nome indica o "verso" do card — é onde a ilustração entra para
# aparecer ao revelar a resposta (Back do Basic, Back Extra/Extra do Cloze).
_ANSWER_FIELD_NAMES = ("back extra", "extra", "back", "verso", "resposta", "answer")


def _answer_field_index(note):
    """Índice do campo onde adicionar a ilustração (o 'verso'). Procura por um
    nome conhecido; se não achar, usa o último campo (que costuma ser o verso)."""
    names = [f["name"].strip().lower() for f in note.note_type()["flds"]]
    for target in _ANSWER_FIELD_NAMES:
        if target in names:
            return names.index(target)
    return len(names) - 1


@socketio.on("generate_illustration")
def handle_generate_illustration():
    """Modo opções: gera uma ilustração didática para o card atual (Nano Banana
    via OpenRouter), passando o livro/assunto (nome do deck) como contexto, e
    envia um preview para o usuário aprovar. A imagem só é gravada no card no
    'approve_illustration'."""
    global pending_illustration
    card = current_card
    if card is None:
        emit("illustration_error", {"message": "Nenhum card ativo"})
        return

    col = get_collection()
    note = card.note()
    deck_id = getattr(card, "odid", 0) or card.did
    book = col.decks.name(deck_id).replace("::", " / ")
    question = current_question or ""
    answer = html2text(strip_images_from_text(answer_only_html(card.answer()))).strip()

    emit("illustration_status", {"message": "Gerando ilustração..."})
    eventlet.sleep(0)
    png = tpool.execute(lambda: generate_illustration(question, answer, book=book))
    if not png:
        emit("illustration_error", {"message": "Não foi possível gerar a imagem."})
        return

    pending_illustration = {"note_id": note.id, "png": png}
    emit("illustration_preview", {
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    })


@socketio.on("approve_illustration")
def handle_approve_illustration():
    """Modo opções: grava a ilustração aprovada na mídia do Anki e a anexa ao
    verso da nota, para o card já aparecer ilustrado nas próximas revisões."""
    global pending_illustration
    if not pending_illustration:
        emit("illustration_error", {"message": "Nenhuma imagem para aprovar."})
        return

    col = get_collection()
    try:
        note = col.get_note(pending_illustration["note_id"])
    except Exception:
        pending_illustration = None
        emit("illustration_error", {"message": "Nota não encontrada."})
        return

    try:
        fname = col.media.write_data("ilustracao.png", pending_illustration["png"])
        idx = _answer_field_index(note)
        sep = "<br>" if note.fields[idx].strip() else ""
        note.fields[idx] += f'{sep}<img src="{fname}">'
        col.update_note(note)
    except Exception as e:
        log.error(f"Erro ao salvar ilustração: {e}")
        emit("illustration_error", {"message": "Erro ao salvar a imagem no card."})
        return
    finally:
        pending_illustration = None

    emit("illustration_saved", {"src": "/media/" + fname})


@socketio.on("discard_illustration")
def handle_discard_illustration():
    """Descarta a ilustração em preview sem gravá-la no card."""
    global pending_illustration
    pending_illustration = None


@socketio.on("generate_video")
def handle_generate_video():
    """Modo opções: gera um vídeo didático curto para o card atual (Veo 3.1 Fast
    via OpenRouter), passando pergunta/resposta e o livro/assunto (nome do deck)
    como contexto, e envia um preview para o usuário aprovar. O vídeo só é gravado
    no card no 'approve_video'. A geração é assíncrona e leva ~1-2 min."""
    global pending_illustration_video
    card = current_card
    if card is None:
        emit("video_error", {"message": "Nenhum card ativo"})
        return

    col = get_collection()
    note = card.note()
    deck_id = getattr(card, "odid", 0) or card.did
    book = col.decks.name(deck_id).replace("::", " / ")
    question = current_question or ""
    answer = html2text(strip_images_from_text(answer_only_html(card.answer()))).strip()

    emit("video_status", {"message": "Gerando vídeo... (pode levar 1-2 min)"})
    eventlet.sleep(0)
    mp4 = tpool.execute(lambda: generate_illustration_video(question, answer, book=book))
    if not mp4:
        emit("video_error", {"message": "Não foi possível gerar o vídeo."})
        return

    pending_illustration_video = {"note_id": note.id, "mp4": mp4}
    emit("video_preview", {
        "video": "data:video/mp4;base64," + base64.b64encode(mp4).decode(),
    })


@socketio.on("approve_video")
def handle_approve_video():
    """Modo opções: grava o vídeo aprovado na mídia do Anki e o anexa ao verso da
    nota como [sound:...], que o Anki renderiza como player de vídeo nas próximas
    revisões."""
    global pending_illustration_video
    if not pending_illustration_video:
        emit("video_error", {"message": "Nenhum vídeo para aprovar."})
        return

    col = get_collection()
    try:
        note = col.get_note(pending_illustration_video["note_id"])
    except Exception:
        pending_illustration_video = None
        emit("video_error", {"message": "Nota não encontrada."})
        return

    try:
        fname = col.media.write_data("ilustracao.mp4", pending_illustration_video["mp4"])
        idx = _answer_field_index(note)
        sep = "<br>" if note.fields[idx].strip() else ""
        note.fields[idx] += f"{sep}[sound:{fname}]"
        col.update_note(note)
    except Exception as e:
        log.error(f"Erro ao salvar vídeo: {e}")
        emit("video_error", {"message": "Erro ao salvar o vídeo no card."})
        return
    finally:
        pending_illustration_video = None

    emit("video_saved", {"src": "/media/" + fname})


@socketio.on("discard_video")
def handle_discard_video():
    """Descarta o vídeo em preview sem gravá-lo no card."""
    global pending_illustration_video
    pending_illustration_video = None


# ---------------------------------------------------------------------------
# Transformar card difícil em questão de múltipla escolha (modo opções)
# ---------------------------------------------------------------------------

def _create_mcq_note(col, source_note, mcq):
    """Cria uma nota de múltipla escolha (MCQ_NOTE_TYPE) a partir do dict gerado
    por generate_mcq, no MESMO deck e com as MESMAS tags da nota original. Os
    textos vêm do LLM em texto plano, então são escapados para não quebrar o HTML
    do campo. Devolve a nova nota (já adicionada à coleção)."""
    model = col.models.by_name(MCQ_NOTE_TYPE)
    if model is None:
        raise RuntimeError(f'Note type "{MCQ_NOTE_TYPE}" não encontrado na coleção')
    note = col.new_note(model)
    field_idx = {f["name"]: i for i, f in enumerate(note.note_type()["flds"])}

    def setf(name, value):
        if name in field_idx:
            note.fields[field_idx[name]] = value

    setf("question", f'<div><strong>{escape(mcq["question"], quote=False)}</strong></div>')
    for letter, text in mcq["options"].items():
        # Mesmo formato dos cards existentes: o texto do campo já traz "A) ...".
        setf(f"option{letter}", f"{letter}) {escape(text, quote=False)}")
    setf("answer", mcq["answer"])
    if mcq.get("note"):
        setf("note", escape(mcq["note"], quote=False))
    for letter, text in (mcq.get("notes") or {}).items():
        setf(f"note{letter}", escape(text, quote=False))

    note.tags = list(source_note.tags)
    orig_cards = source_note.cards()
    if orig_cards:
        deck_id = getattr(orig_cards[0], "odid", 0) or orig_cards[0].did
    else:
        deck_id = col.decks.get_current_id()
    col.add_note(note, deck_id)
    return note


@socketio.on("make_mcq")
def handle_make_mcq():
    """Modo opções: gera uma questão de múltipla escolha a partir do card atual
    (Claude Sonnet via OpenRouter), citando o baralho/assunto como referência no
    enunciado, e envia um preview. A criação/substituição só ocorre no
    'approve_mcq' — assim o usuário confere antes de apagar o card original."""
    global pending_mcq
    card = current_card
    if card is None:
        emit("mcq_error", {"message": "Nenhum card ativo"})
        return

    col = get_collection()
    note = card.note()
    deck_id = getattr(card, "odid", 0) or card.did
    book = col.decks.name(deck_id).replace("::", " / ")
    question = current_question or html2text(strip_images_from_text(card.question())).strip()
    answer = html2text(strip_images_from_text(answer_only_html(card.answer()))).strip()

    emit("mcq_status", {"message": "Gerando múltipla escolha..."})
    eventlet.sleep(0)
    mcq = tpool.execute(lambda: generate_mcq(question, answer, book=book))
    if not mcq:
        emit("mcq_error", {"message": "Não foi possível gerar a múltipla escolha."})
        return

    pending_mcq = {"note_id": note.id, "mcq": mcq}
    emit("mcq_preview", {"mcq": mcq})


@socketio.on("approve_mcq")
def handle_approve_mcq():
    """Modo opções: cria o card de múltipla escolha e APAGA a nota original
    (transformação). Cria primeiro e só então remove — se a criação falhar, o
    card original é preservado. Depois avança (lista ou próximo card)."""
    global pending_mcq, current_card
    if not pending_mcq:
        emit("mcq_error", {"message": "Nenhuma múltipla escolha para criar."})
        return

    col = get_collection()
    try:
        source = col.get_note(pending_mcq["note_id"])
    except Exception:
        pending_mcq = None
        emit("mcq_error", {"message": "Nota original não encontrada."})
        return

    try:
        _create_mcq_note(col, source, pending_mcq["mcq"])
        col.remove_notes([source.id])
    except Exception as e:
        log.error(f"Erro ao transformar em múltipla escolha (nota {source.id}): {e}")
        emit("mcq_error", {"message": "Erro ao criar o card de múltipla escolha."})
        return
    finally:
        pending_mcq = None

    current_card = None
    emit("mcq_created", {"message": "Card transformado em múltipla escolha."})
    if session_flow == "list":
        emit_card_list()
    else:
        send_next_card()


@socketio.on("discard_mcq")
def handle_discard_mcq():
    """Descarta a múltipla escolha em preview sem criar nada nem apagar o card."""
    global pending_mcq
    pending_mcq = None


@socketio.on("skip_card")
def handle_skip_card():
    """Pula o card atual enterrando-o (bury). Apenas "esconder na sessão" não
    funciona: o card continuaria no topo da fila do Anki e bloquearia
    ('not at top of queue') os cards atrás dele. Bury tira o card da frente da
    fila sem mudar intervalo/ease, e o Anki o desenterra no dia seguinte."""
    if current_card:
        get_collection().sched.bury_cards([current_card.id])
        record_skip()
    if session_flow == "list":
        emit_card_list()
    else:
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
    if session_flow == "list":
        emit_card_list()
    else:
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
