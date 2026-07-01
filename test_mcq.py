"""Teste da transformação "card difícil → múltipla escolha".

Cobre as duas partes que mexem na coleção (o resto é UI/socket):

  1. anki_ai._parse_mcq_json: normaliza o JSON do LLM (letras contíguas a partir
     de 'A', gabarito remapeado, formas dict e lista, e rejeição de lixo).
  2. anki_web._create_mcq_note: monta a nota IKKZ__MCQ_26.PT_BR.NATIVE com os
     campos certos (question em <strong>, optionX = "X) ...", answer = letra,
     note/noteX), no mesmo deck e tags da nota original — e a substituição
     (criar a MCQ e remover a original) deixa a coleção coerente.

Roda numa coleção temporária (não toca na real) e sem APIs externas.
"""
import os
import tempfile

os.environ.setdefault("OPENAI_API_KEY", "x")

from anki.collection import Collection

import anki_web
from anki_web import MCQ_NOTE_TYPE, _create_mcq_note
from anki_ai import _parse_mcq_json


def _build_mcq_model(col):
    """Recria o note type de múltipla escolha (campos que _create_mcq_note usa)."""
    mm = col.models
    m = mm.new(MCQ_NOTE_TYPE)
    for fname in ["question", "optionA", "optionB", "optionC", "optionD", "optionE",
                  "answer", "note", "noteA", "noteB", "noteC", "noteD", "noteE"]:
        mm.add_field(m, mm.new_field(fname))
    tmpl = mm.new_template("Card 1")
    tmpl["qfmt"] = "{{question}}"
    tmpl["afmt"] = "{{answer}}"
    mm.add_template(m, tmpl)
    mm.add(m)
    return mm.by_name(MCQ_NOTE_TYPE)


def test_parser():
    a = _parse_mcq_json(
        '{"question":"De acordo com o Arte Naval, o que X?",'
        '"options":{"A":"aa","B":"bb","C":"cc","D":"dd"},"answer":"B",'
        '"note":"pq bb","notes":{"A":"na","B":"ok","C":"nc","D":"nd"}}'
    )
    assert a and a["answer"] == "B" and a["options"]["B"] == "bb", a
    assert a["notes"]["B"] == "ok"

    # Letras com buraco (A,B,D) → recontíguas A,B,C e gabarito D→C.
    b = _parse_mcq_json('{"question":"q","options":{"A":"x","B":"y","D":"z"},"answer":"D"}')
    assert b and list(b["options"]) == ["A", "B", "C"] and b["answer"] == "C", b

    # Forma de lista, gabarito por letra.
    c = _parse_mcq_json('{"question":"q","options":["p","q","r","s"],"answer":"C"}')
    assert c and c["options"]["C"] == "r", c

    # Inválidos → None.
    assert _parse_mcq_json('{"question":"q","options":{"A":"x","B":"y"},"answer":"Z"}') is None
    assert _parse_mcq_json('{"options":{"A":"x","B":"y"},"answer":"A"}') is None
    assert _parse_mcq_json("não sei responder") is None
    print("OK — _parse_mcq_json normaliza e valida")


def test_create_and_replace():
    tmp = tempfile.mkdtemp()
    col = Collection(os.path.join(tmp, "col.anki2"))
    anki_web.collection = col
    _build_mcq_model(col)

    # Nota original (Basic) num deck próprio e com uma tag.
    basic = col.models.by_name("Basic") or col.models.by_name("Básico")
    did = col.decks.id("Arte Naval::Cap 3")
    src = col.new_note(basic)
    src.fields[0] = "O que as cartas meteorológicas mostram?"
    src.fields[1] = "Isóbaras, centros de pressão, frentes e tempestades."
    src.tags = ["nautica"]
    col.add_note(src, did)
    src_id = src.id

    mcq = {
        "question": "De acordo com o Arte Naval, o que as cartas meteorológicas mostram?",
        "options": {"A": "Só o vento", "B": "Isóbaras e frentes", "C": "Só nuvens", "D": "Marés"},
        "answer": "B",
        "note": "B reproduz a resposta do flashcard.",
        "notes": {"A": "incompleta", "B": "correta", "C": "incompleta", "D": "fora do tema"},
    }

    note = _create_mcq_note(col, src, mcq)
    col.remove_notes([src.id])  # a transformação apaga a original

    got = col.get_note(note.id)
    fields = {f["name"]: got.fields[i] for i, f in enumerate(got.note_type()["flds"])}
    assert fields["question"] == "<div><strong>De acordo com o Arte Naval, o que as cartas meteorológicas mostram?</strong></div>", fields["question"]
    assert fields["optionA"] == "A) Só o vento", fields["optionA"]
    assert fields["optionB"] == "B) Isóbaras e frentes", fields["optionB"]
    assert fields["answer"] == "B", fields["answer"]
    assert fields["note"] == "B reproduz a resposta do flashcard."
    assert fields["noteB"] == "correta" and fields["noteD"] == "fora do tema"
    assert list(got.tags) == ["nautica"], got.tags

    # Foi para o mesmo deck da original e a original sumiu.
    assert got.note_type()["name"] == MCQ_NOTE_TYPE
    assert col.decks.name(got.cards()[0].did) == "Arte Naval::Cap 3"
    assert not col.find_notes(f"nid:{src_id}"), "nota original deveria ter sido apagada"

    # Escapa HTML do texto do LLM (não quebra o campo).
    mcq2 = dict(mcq, question="x < y & z", options={"A": "a<b", "B": "ok"}, answer="B", notes={})
    n2 = _create_mcq_note(col, got, mcq2)
    f2 = {f["name"]: col.get_note(n2.id).fields[i] for i, f in enumerate(col.get_note(n2.id).note_type()["flds"])}
    assert "&lt;" in f2["question"] and "&amp;" in f2["question"], f2["question"]
    assert f2["optionA"] == "A) a&lt;b", f2["optionA"]

    print("OK — _create_mcq_note monta campos, copia deck/tags e substitui a nota")


if __name__ == "__main__":
    test_parser()
    test_create_and_replace()
