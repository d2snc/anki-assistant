"""Teste do modo "Responder Listagem" (anki_web.answer_specific_card).

O scheduler v3 do Anki só deixa responder o card no TOPO da fila (responder um
review/novo mais fundo dá `not at top of queue` — ver memória v3-answer-top-of-
queue-rule). O modo listagem deixa o usuário escolher QUALQUER card da pilha para
responder; para isso, answer_specific_card enterra temporariamente a frente da
fila, responde o card escolhido e desenterra de volta SÓ o que enterrou.

Este teste garante, numa coleção temporária (não toca na real):

  1. dá pra responder um Due, um New e um Learn fundos, fora de ordem, sem erro;
  2. os demais cards da fila voltam intactos (mesma quantidade por coluna, menos
     o card respondido) — o bury/unbury não some com nada nem mexe no resto.

Roda sem APIs externas: monta a coleção e chama as funções direto.
"""
import os
import time
import tempfile

os.environ.setdefault("OPENAI_API_KEY", "x")  # evita exigir credenciais no import

from anki.collection import Collection

import anki_web


def _build_collection(path):
    """6 revisões (Due), 4 novos e 3 learning intradiário, todos estudáveis hoje."""
    col = Collection(path)
    model = col.models.by_name("Basic") or col.models.by_name("Básico")
    today = col.sched.today

    def add(front):
        note = col.new_note(model)
        note.fields[0] = front
        note.fields[1] = "resposta " + front
        col.add_note(note, col.decks.id("Default"))
        return note.cards()[0]

    revs = []
    for i in range(6):
        c = add(f"Pergunta rev{i}")
        c.queue = 2; c.type = 2; c.due = today; c.ivl = 5
        col.update_card(c); revs.append(c.id)

    news = [add(f"Pergunta new{i}").id for i in range(4)]

    lrns = []
    for i in range(3):
        c = add(f"Pergunta lrn{i}")
        # due no passado (epoch realista) => learning "vencido", na frente da fila
        c.queue = 1; c.type = 1; c.due = int(time.time()) - 50 + i; c.left = 1001
        col.update_card(c); lrns.append(c.id)

    return col, revs, news, lrns


def _sizes(col):
    g = anki_web.collect_queue_by_state(col)
    return {k: len(v) for k, v in g.items()}


def main():
    tmp = tempfile.mkdtemp()
    col, revs, news, lrns = _build_collection(os.path.join(tmp, "col.anki2"))
    anki_web.collection = col  # injeta na get_collection do módulo

    base = _sizes(col)
    assert base == {"new": 4, "learn": 3, "review": 6}, base

    # Preview da listagem deve trazer o texto da pergunta (sem HTML).
    groups = anki_web.collect_queue_by_state(col)
    assert any("rev" in c["preview"] for c in groups["review"]), groups["review"]

    # 1) Responde um Due FUNDO (5º) fora de ordem.
    assert anki_web.answer_specific_card(col, revs[4], 3) is not None, "Due fundo falhou"
    s = _sizes(col)
    assert s["review"] == 5 and s["new"] == 4, ("Due: resto não intacto", base, s)
    assert revs[4] not in [c["id"] for c in anki_web.collect_queue_by_state(col)["review"]]

    # 2) Responde um New FUNDO (4º) fora de ordem.
    assert anki_web.answer_specific_card(col, news[3], 3) is not None, "New fundo falhou"
    assert _sizes(col)["new"] == 3, "New não saiu da pilha"

    # 3) Responde um Learning FUNDO (3º) fora de ordem (só o topo é respondível,
    #    então enterra tudo à frente). Não deve dar erro nem sumir com os outros.
    assert anki_web.answer_specific_card(col, lrns[2], 3) is not None, "Learn fundo falhou"
    final = _sizes(col)
    # Sobra: a fila ainda existe e nada além do respondido sumiu de review/new.
    assert final["review"] == 5 and final["new"] == 3, ("Learn mexeu no resto", final)

    # 4) Card inexistente / já respondido: devolve None sem estourar.
    assert anki_web.answer_specific_card(col, revs[4], 3) is None

    print("OK — Due/New/Learn respondidos fora de ordem; resto da fila intacto")


def test_deep_card_beyond_old_window():
    """Pilha grande (>500): responder um card FUNDO tem que tirá-lo da lista.

    A lista é montada com a janela inteira (QUEUE_FETCH_LIMIT), então o usuário
    pode escolher um card além da posição 500. answer_specific_card precisa usar a
    MESMA janela; com a antiga (500) o card "não saía" da lista (regressão)."""
    tmp = tempfile.mkdtemp()
    col = Collection(os.path.join(tmp, "col.anki2"))
    model = col.models.by_name("Basic") or col.models.by_name("Básico")
    did = col.decks.id("Default")

    # 700 learning intradiário (não sofrem limite diário, então enchem a fila).
    base = int(time.time()) - 5000
    ids = []
    for i in range(700):
        note = col.new_note(model)
        note.fields[0] = f"Pergunta {i:04d}"
        note.fields[1] = "resposta"
        col.add_note(note, did)
        c = note.cards()[0]
        c.queue = 1; c.type = 1; c.due = base + i; c.left = 1001
        col.update_card(c); ids.append(c.id)

    anki_web.collection = col
    learn = anki_web.collect_queue_by_state(col)["learn"]
    target = ids[640]  # bem além da antiga janela de 500
    assert target in [c["id"] for c in learn], "card fundo deveria estar na lista"

    assert anki_web.answer_specific_card(col, target, 4) is not None, "card fundo falhou"
    final = anki_web.collect_queue_by_state(col)
    still = target in [c["id"] for col_cards in final.values() for c in col_cards]
    assert not still, "card respondido (Easy) deveria ter saído da lista"
    assert len(final["learn"]) == 699, ("resto da pilha não intacto", len(final["learn"]))

    print("OK — card fundo (pos 640, além da janela antiga) sai da lista ao responder")


if __name__ == "__main__":
    main()
    test_deep_card_beyond_old_window()
