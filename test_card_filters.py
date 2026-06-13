"""Testes das funções puras de limpeza de card (anki_web).

Cobre answer_only_html — tira a pergunta (FrontSide) repetida na resposta do
Basic. Não toca no Anki nem em APIs externas.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "x")  # evita exigir credenciais no import

import anki_web


def test_answer_only_html():
    # Resposta renderizada do Basic: pergunta + <hr id=answer> + resposta.
    full = "Capital da França?<hr id=answer>Paris"
    assert anki_web.answer_only_html(full) == "Paris", "deve remover a pergunta antes do hr"

    # Variação com aspas no id e atributos extras.
    full2 = 'Quem escreveu Dom Casmurro?<hr id="answer" class="x">Machado de Assis'
    assert anki_web.answer_only_html(full2) == "Machado de Assis"

    # Sem o hr (note type customizado): devolve o HTML inteiro como fallback.
    plain = "<p>Resposta sem separador</p>"
    assert anki_web.answer_only_html(plain) == plain

    print("OK — answer_only_html")


def main():
    test_answer_only_html()
    print("OK — testes de limpeza de card passaram")


if __name__ == "__main__":
    main()
