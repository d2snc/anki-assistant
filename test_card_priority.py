"""Teste da priorização de cards na sessão de voz (anki_web._pick_next_card).

Reproduz, numa coleção Anki temporária (não toca na real), o cenário que gerava
os dois bugs relatados:

  * cards de learning intradiário empilhados na FRENTE de muitas revisões (Due);
  * com fetch_limit=1 o app só via o topo -> servia learning, o Due nunca fechava
    e o contador de Due nunca caía.

Garante dois invariantes do scheduler v3 que descobrimos empiricamente:

  1. Nunca tomamos 'not at top of queue' (InvalidInput) ao responder — só pulamos
     a frente de learning para o PRIMEIRO não-learning, nunca reordenamos entre os
     não-learning.
  2. Nenhum card de learning é servido enquanto houver um Due/New alcançável
     ( "fazer o Due antes do Learn").

Roda sem APIs externas: monta a coleção e dirige advance_card direto.
"""
import os
import shutil
import tempfile

os.environ.setdefault("OPENAI_API_KEY", "x")  # evita exigir credenciais no import

from anki.collection import Collection

import anki_web


def _build_collection(path):
    """Coleção com 7 learning intradiário na frente, 15 revisões e 8 novos."""
    col = Collection(path)
    model = col.models.by_name("Basic") or col.models.by_name("Básico")
    today = col.sched.today  # dia interno do Anki (para due de revisão)
    now = 1  # due baixo => learning "vencido" e na frente da fila

    def add(front):
        note = col.new_note(model)
        note.fields[0] = front
        note.fields[1] = "resposta " + front
        col.add_note(note, col.decks.id("Default"))
        return note.cards()[0]

    # Revisões (queue/type=2), vencidas hoje.
    for i in range(15):
        c = add(f"rev{i}")
        c.queue = 2
        c.type = 2
        c.due = today
        c.ivl = 5
        col.update_card(c)

    # Learning intradiário (queue/type=1), vencido (due no passado) -> vai pra frente.
    for i in range(7):
        c = add(f"lrn{i}")
        c.queue = 1
        c.type = 1
        c.due = now + i
        c.left = 1001
        col.update_card(c)

    # Novos (queue 0) ficam como estão.
    for i in range(8):
        add(f"new{i}")

    return col


def main():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "collection.anki2")
    col = _build_collection(path)

    # Injeta a coleção de teste no módulo (substitui get_collection).
    anki_web.collection = col
    _orig_get = anki_web.get_collection
    anki_web.get_collection = lambda: col

    try:
        n0, l0, r0 = col.sched.counts()
        assert (n0, l0, r0) == (8, 7, 15), f"contagem inicial inesperada: {(n0, l0, r0)}"

        served = []
        invalid = 0
        ordering_violations = 0
        guard = 0
        while True:
            guard += 1
            assert guard < 500, "loop não terminou — possível travamento"
            res = anki_web.advance_card()
            if res is None:
                break
            card, _q, _a = res
            state = anki_web.card_state(card)

            # Invariante 2: ao servir learning, não pode haver não-learning alcançável.
            if state == "learn":
                reachable = anki_web.get_prioritized_card(
                    col, fetch_limit=anki_web.CARD_LOOKAHEAD,
                    predicate=lambda c: getattr(c, "queue", None)
                    != anki_web.QUEUE_LEARN_INTRADAY,
                )
                if reachable is not None:
                    ordering_violations += 1

            # Invariante 1: responder nunca deve estourar 'not at top of queue'.
            try:
                col.sched.answerCard(card, 3)  # sempre "acertou"
            except Exception as e:  # noqa: BLE001
                invalid += 1
                raise AssertionError(
                    f"InvalidInput ao responder {state} card {card.id}: {e!r}"
                )
            anki_web.current_card = None
            served.append(state)

        assert invalid == 0
        assert ordering_violations == 0, (
            f"{ordering_violations}x serviu learning com Due/New ainda alcançável"
        )
        assert col.sched.counts() == (0, 0, 0), "a fila deveria ter esvaziado"

        # Todas as 15 revisões e os 8 novos saíram ANTES de qualquer learning
        # (acerto não regenera learning, então a ordem é limpa).
        first_learn = next((i for i, s in enumerate(served) if s == "learn"), len(served))
        assert all(s != "learn" for s in served[:first_learn])
        assert served[:first_learn].count("review") == 15, served[:first_learn]
        assert served[:first_learn].count("new") == 8, served[:first_learn]

        print("OK — Due/New servidos antes do Learn, sem 'not at top of queue'")
        print("ordem servida:", " ".join(served))
    finally:
        anki_web.get_collection = _orig_get
        col.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
