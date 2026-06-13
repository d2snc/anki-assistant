"""Teste do rastreamento de estatísticas/histórico por sessão (anki_web).

Exercita as funções puras (sem precisar do Anki nem dos modelos): início de
sessão, contagem de acertos/erros/pulos, finalização com cálculo de métricas,
persistência em JSON e agregação da rota /history.
"""
import os
import json
import tempfile

import anki_web


def main():
    # Aponta o histórico para um arquivo temporário (não toca no real).
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)  # começa inexistente, como em uso real
    anki_web.SESSIONS_PATH = tmp.name

    try:
        # Histórico vazio ainda não existe em disco.
        assert anki_web.load_history() == [], "histórico inicial deve ser vazio"

        # ---- Sessão 1: 3 respostas (2 certas, 1 errada) + 1 pulo ----
        anki_web.start_session_stats("Inglês::Vocabulário")
        anki_web.session_stats["started_at"] -= 65  # simula 65s decorridos
        anki_web.record_answer(passed=True, score=4)
        anki_web.record_answer(passed=True, score=3)
        anki_web.record_answer(passed=False, score=1)
        anki_web.record_skip()
        rec = anki_web.finalize_session()

        assert rec is not None, "deve retornar o resumo da sessão"
        assert rec["deck"] == "Inglês::Vocabulário"
        assert rec["answered"] == 3
        assert rec["correct"] == 2
        assert rec["wrong"] == 1
        assert rec["skipped"] == 1
        assert rec["accuracy"] == round(100 * 2 / 3, 1), rec["accuracy"]
        assert rec["avg_score"] == round((4 + 3 + 1) / 3, 2), rec["avg_score"]
        assert rec["duration_sec"] >= 60, rec["duration_sec"]
        assert anki_web.session_stats is None, "sessão deve zerar após finalizar"

        # finalize idempotente: segunda chamada não duplica registro.
        assert anki_web.finalize_session() is None

        # ---- Sessão vazia não é registrada ----
        anki_web.start_session_stats("Deck vazio")
        assert anki_web.finalize_session() is None, "sessão sem atividade não grava"

        # ---- Sessão 2: só 1 acerto ----
        anki_web.start_session_stats("Náutica")
        anki_web.record_answer(passed=True, score=4)
        anki_web.finalize_session()

        # ---- Persistência em disco ----
        with open(tmp.name, encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 2, f"esperado 2 sessões gravadas, veio {len(saved)}"
        assert saved[0]["deck"] == "Inglês::Vocabulário"
        assert saved[1]["deck"] == "Náutica"

        # ---- Rota /history: mais recentes primeiro + totais agregados ----
        with anki_web.app.test_client() as client:
            resp = client.get("/history")
            assert resp.status_code == 200
            data = resp.get_json()

        sessions = data["sessions"]
        totals = data["totals"]
        assert sessions[0]["deck"] == "Náutica", "mais recente deve vir primeiro"
        assert totals["sessions"] == 2
        assert totals["answered"] == 4  # 3 + 1
        assert totals["correct"] == 3   # 2 + 1
        assert totals["accuracy"] == round(100 * 3 / 4, 1), totals["accuracy"]

        print("OK — todos os testes de histórico de sessão passaram")
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
