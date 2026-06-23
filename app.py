"""
CodeLens — Streamlit UI
Запуск: streamlit run app.py
"""

import json
import time
import os
import sys

import streamlit as st

# ──────────────────────────────────────────────
# Page config (первый вызов st в файле)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CodeLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Lazy imports с человеческой ошибкой
# ──────────────────────────────────────────────
try:
    from ml_core.db_client import ChromaDBClient
    from ml_core.models import EmbeddingModelRegistry
    from backend.app.controllers.rag_controller import RAGController
    from ml_core.llm_client import LLMClient, check_ollama_available, OPENROUTER_MODELS
except ImportError as e:
    st.error(f"Ошибка импорта: {e}\nУбедись, что запускаешь из корня проекта: `streamlit run app.py`")
    st.stop()

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Основная палитра */
:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --accent: #7c6af7;
    --accent2: #4fd1c5;
    --text: #e2e8f0;
    --muted: #718096;
    --success: #48bb78;
    --warn: #ed8936;
    --radius: 10px;
}

/* Карточка результата */
.chunk-card {
    background: var(--surface);
    border: 1px solid var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 14px;
}
.chunk-card:hover { border-left-color: var(--accent2); }

/* Заголовок карточки */
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
    gap: 6px;
}
.chunk-name {
    font-weight: 700;
    font-size: 15px;
    color: var(--accent2);
    font-family: 'JetBrains Mono', monospace;
}
.chunk-type-badge {
    background: var(--surface2);
    color: var(--muted);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.chunk-file {
    color: var(--muted);
    font-size: 12px;
    font-family: monospace;
    margin-bottom: 6px;
}
.chunk-score {
    color: var(--success);
    font-weight: 700;
    font-size: 13px;
    white-space: nowrap;
}
.chunk-docstring {
    color: var(--muted);
    font-size: 12px;
    font-style: italic;
    margin-bottom: 8px;
    border-left: 2px solid var(--surface2);
    padding-left: 8px;
}

/* LLM ответ */
.llm-answer {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 18px 22px;
    margin-top: 18px;
    color: var(--text);
    line-height: 1.7;
}
.llm-answer h4 { color: var(--accent2); margin-bottom: 10px; }

/* Статистика */
.stat-box {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 12px 16px;
    text-align: center;
    border: 1px solid var(--surface2);
}
.stat-value { font-size: 24px; font-weight: 800; color: var(--accent); }
.stat-label { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* Метрика Precision */
.precision-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 13px;
}
.precision-row:nth-child(odd) { background: var(--surface); }
.p-qid { color: var(--muted); font-family: monospace; }
.p-score-high { color: var(--success); font-weight: 700; }
.p-score-mid  { color: var(--warn); font-weight: 700; }
.p-score-low  { color: #fc8181; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "controller": None,
        "llm_client": None,
        "search_results": [],
        "last_query": "",
        "llm_answer": "",
        "llm_loading": False,
        "search_time": 0.0,
        "chat_history": [],  # [(query, answer), ...]
        "precision_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────
# Cached controller
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def get_controller(model_key: str, db_path: str) -> RAGController:
    ctrl = RAGController()
    ctrl.embedder.set_active_model(model_key)
    return ctrl


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 CodeLens")
    st.caption("RAG-поиск по кодовой базе")
    st.divider()

    # Навигация
    page = st.radio(
        "Раздел",
        ["Поиск", "Индексация", "Precision@5"],
        label_visibility="collapsed",
    )

    st.divider()

    # Embedding model
    st.markdown("**Модель эмбеддингов**")
    emb_model = st.selectbox(
        "emb_model",
        options=["paraphrase-multilingual", "bge-m3"],
        format_func=lambda x: {
            "paraphrase-multilingual": "MiniLM-L12 (быстро)",
            "bge-m3": "BGE-M3 (точнее)",
        }[x],
        label_visibility="collapsed",
    )

    # DB path
    db_path = st.text_input("Путь к ChromaDB", value="data/vector_db")

    # Top-K
    top_k = st.slider("Результатов (Top-K)", 1, 10, 5)

    st.divider()

    # LLM settings
    st.markdown("**LLM для ответов**")
    llm_enabled = st.toggle("Включить LLM-режим", value=False)

    llm_provider = st.radio(
        "Провайдер",
        ["Ollama (локально)", "OpenRouter (облачно)"],
        disabled=not llm_enabled,
    )

    ollama_model = "mistral:7b"
    openrouter_key = ""
    openrouter_model_label = "mistral-7b (free)"

    if llm_enabled:
        if "Ollama" in llm_provider:
            ollama_model = st.selectbox(
                "Модель Ollama",
                ["mistral:7b", "llama3:8b", "gemma:7b", "codellama:7b"],
            )
            ollama_ok = check_ollama_available()
            if ollama_ok:
                st.success("Ollama запущен ✓")
            else:
                st.warning("Ollama не найден. Запусти: `ollama serve`")
        else:
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                placeholder="sk-or-...",
                value=os.getenv("OPENROUTER_API_KEY", ""),
            )
            openrouter_model_label = st.selectbox(
                "Модель",
                list(OPENROUTER_MODELS.keys()),
            )

    # Применить настройки
    if st.button("Применить настройки", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.controller = None
        st.session_state.llm_client = None
        st.rerun()

    st.divider()
    st.caption("Чемпионат Ростелеком «Топ Уровень» | 2026")


# ──────────────────────────────────────────────
# Инициализация контроллера и LLM
# ──────────────────────────────────────────────
if st.session_state.controller is None:
    try:
        st.session_state.controller = get_controller(emb_model, db_path)
    except Exception as e:
        st.error(f"Ошибка инициализации: {e}")
        st.stop()

ctrl: RAGController = st.session_state.controller

if llm_enabled and st.session_state.llm_client is None:
    provider = "ollama" if "Ollama" in llm_provider else "openrouter"
    st.session_state.llm_client = LLMClient(
        provider=provider,
        ollama_model=ollama_model,
        openrouter_api_key=openrouter_key,
        openrouter_model_label=openrouter_model_label,
    )
elif not llm_enabled:
    st.session_state.llm_client = None


# ══════════════════════════════════════════════
# PAGE: ПОИСК
# ══════════════════════════════════════════════
if page == "Поиск":

    st.markdown("## Поиск по кодовой базе")
    st.caption("Семантический поиск на русском и английском языке")

    # Поисковая строка
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "query",
            placeholder="например: как создаётся токен доступа? / how does the project handle auth errors?",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Найти 🔍", use_container_width=True, type="primary")

    # Быстрые примеры
    st.markdown("**Попробуй:**")
    examples = [
        "как создаётся токен доступа?",
        "how does password hashing work?",
        "где обрабатываются ошибки авторизации?",
        "database session management",
    ]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        with col:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                query = ex
                search_clicked = True

    # ── Поиск ──
    if search_clicked and query.strip():
        st.session_state.last_query = query.strip()
        st.session_state.llm_answer = ""

        with st.spinner("Ищу..."):
            t0 = time.time()
            try:
                results = ctrl.find_relevant_code(query.strip())
            except Exception as e:
                st.error(f"Ошибка поиска: {e}")
                results = []
            st.session_state.search_time = time.time() - t0
            st.session_state.search_results = results

        # LLM ответ (если включён)
        if llm_enabled and st.session_state.llm_client and results:
            with st.spinner("Генерирую ответ..."):
                try:
                    answer = st.session_state.llm_client.generate(query.strip(), results)
                    st.session_state.llm_answer = answer
                    st.session_state.chat_history.append((query.strip(), answer))
                except Exception as e:
                    st.session_state.llm_answer = f"Ошибка LLM: {e}"

    # ── Результаты ──
    results = st.session_state.search_results
    if results:
        # Статистика
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(results)}</div><div class="stat-label">результатов</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{st.session_state.search_time:.2f}s</div><div class="stat-label">время ответа</div></div>', unsafe_allow_html=True)
        with sc3:
            unique_files = len({r.get("file_path", "") for r in results})
            st.markdown(f'<div class="stat-box"><div class="stat-value">{unique_files}</div><div class="stat-label">файлов</div></div>', unsafe_allow_html=True)

        st.divider()

        # LLM ответ (отображение)
        if st.session_state.llm_answer:
            st.markdown(f"""<div class="llm-answer">
<h4>💬 Ответ ассистента</h4>
{st.session_state.llm_answer}
</div>""", unsafe_allow_html=True)
            st.divider()

        # Карточки результатов
        st.markdown(f"### Найденные фрагменты кода")
        for i, chunk in enumerate(results):
            file_path = chunk.get("file_path", "unknown")
            name = chunk.get("name", "unknown")
            el_type = chunk.get("element_type", "function")
            start = chunk.get("start_line", 1)
            end = chunk.get("end_line", 1)
            docstring = chunk.get("docstring", "") or ""
            content = chunk.get("content", "")

            # Убираем enriched-префикс если есть
            for prefix_key in ["Файл:", "File path", "Сущность:", "Код:"]:
                if content.startswith(prefix_key) or f"\n{prefix_key}" in content:
                    lines = content.split("\n")
                    code_lines = [l for l in lines if not any(l.startswith(k) for k in ["Файл:", "Сущность:", "Код:"])]
                    content = "\n".join(code_lines).strip()
                    break

            # Нормализуем score: если это float score (final_score), используем его
            raw_score = chunk.get("final_score", None)
            if raw_score is not None:
                score_pct = min(100, int(raw_score * 40))  # нормализация к %
            else:
                score_pct = 0

            # Карточка
            st.markdown(f"""
<div class="chunk-card">
  <div class="chunk-header">
    <div>
      <span class="chunk-name">{name}</span>
      <span class="chunk-type-badge">{el_type}</span>
    </div>
    <span class="chunk-score">релевантность {score_pct}%</span>
  </div>
  <div class="chunk-file">📄 {file_path} · строки {start}–{end}</div>
  {f'<div class="chunk-docstring">📝 {docstring[:200]}{"…" if len(docstring) > 200 else ""}</div>' if docstring else ""}
</div>
""", unsafe_allow_html=True)

            st.code(content, language="python")

    elif st.session_state.last_query:
        st.info("По запросу ничего не найдено. Попробуй другую формулировку или сначала проиндексируй проект.")

    # История чата
    if st.session_state.chat_history and llm_enabled:
        with st.expander(f"📜 История диалога ({len(st.session_state.chat_history)} запросов)"):
            for q, a in reversed(st.session_state.chat_history[-10:]):
                st.markdown(f"**🙋 {q}**")
                st.markdown(a)
                st.divider()
        if st.button("Очистить историю"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════
# PAGE: ИНДЕКСАЦИЯ
# ══════════════════════════════════════════════
elif page == "Индексация":
    st.markdown("## Индексация проекта")

    col1, col2 = st.columns([3, 1])
    with col1:
        project_path = st.text_input(
            "Путь к директории с кодом",
            value="gymhero",
            placeholder="gymhero / /path/to/project",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_index = st.button("Индексировать ▶", type="primary", use_container_width=True)

    # Информация о текущей базе
    try:
        indexed_files = ctrl.get_indexed_files()
        if indexed_files:
            st.success(f"В базе уже {len(indexed_files)} файлов")
            with st.expander("Показать проиндексированные файлы"):
                for f in sorted(indexed_files):
                    st.text(f)
        else:
            st.warning("База пуста. Запусти индексацию.")
    except Exception as e:
        st.error(f"Не удалось получить список файлов: {e}")

    if run_index:
        if not os.path.exists(project_path):
            st.error(f"Путь не найден: `{project_path}`")
        else:
            progress = st.progress(0, text="Начало индексации...")
            log_box = st.empty()

            try:
                from ml_core.pipeline import IndexingPipeline
                pipeline = IndexingPipeline(
                    db_client=ctrl.db_client,
                    embedder=ctrl.embedder,
                )
                progress.progress(20, "Парсинг файлов...")
                stats = pipeline.index_project(project_path)
                progress.progress(100, "Готово!")

                st.balloons()
                st.success("Индексация завершена!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Файлов найдено", stats.get("total_files", 0))
                c2.metric("Обработано", stats.get("processed_files", 0))
                c3.metric("Чанков", stats.get("chunks_count", 0))

                langs = stats.get("languages", {})
                if langs:
                    st.markdown("**Языки:**")
                    for lang, cnt in langs.items():
                        st.markdown(f"- `{lang}`: {cnt} чанков")

            except Exception as e:
                progress.empty()
                st.error(f"Ошибка индексации: {e}")


# ══════════════════════════════════════════════
# PAGE: PRECISION@5
# ══════════════════════════════════════════════
elif page == "Precision@5":
    st.markdown("## Метрика качества — Precision@5")
    st.caption("Запускает оценку по тестовому набору eval_questions.json")

    col_q, col_r = st.columns(2)
    with col_q:
        questions_path = st.text_input("eval_questions.json", value="eval_questions.json")
    with col_r:
        results_path = st.text_input("results.json (куда сохранить)", value="results.json")

    run_eval = st.button("Запустить оценку ▶", type="primary")

    if run_eval:
        if not os.path.exists(questions_path):
            st.error(f"Файл не найден: {questions_path}")
        else:
            with st.spinner("Обрабатываю вопросы..."):
                try:
                    from backend.app.services.generate_submision import generate_submission
                    from backend.app.services.search_service import SearchService

                    search_svc = SearchService(ctrl.db_client, ctrl.embedder)
                    generate_submission(
                        project_name=ctrl.project_name,
                        questions_json_path=questions_path,
                        output_submission_path=results_path,
                        search_service=search_svc,
                    )

                    # Считаем Precision@5
                    with open(questions_path, "r", encoding="utf-8") as f:
                        questions = json.load(f)
                    with open(results_path, "r", encoding="utf-8") as f:
                        predictions = json.load(f)

                    # Упрощённый scoring (без ±2 tolerance, для UI)
                    pred_index = {p["question_id"]: p["top_5_chunks"] for p in predictions}
                    rows = []
                    scores = []
                    for q in questions:
                        qid = q.get("question_id", "")
                        correct = q.get("correct_chunk_ids", [])
                        preds = pred_index.get(qid, [])
                        matched = sum(1 for p in preds if p in correct)
                        s = matched / min(5, len(correct)) if correct else 0
                        scores.append(s)
                        rows.append({
                            "qid": qid,
                            "query": q.get("query", "")[:60],
                            "difficulty": q.get("difficulty", ""),
                            "lang": q.get("language", ""),
                            "score": s,
                            "matched": matched,
                            "total": len(correct),
                        })

                    mean_p5 = sum(scores) / len(scores) if scores else 0
                    target_ok = mean_p5 >= 0.6

                    # Итог
                    m1, m2, m3 = st.columns(3)
                    color = "success" if target_ok else "warning"
                    m1.metric("Mean Precision@5", f"{mean_p5:.3f}", delta="≥0.6 ✓" if target_ok else "< цели")
                    m2.metric("Вопросов", len(rows))
                    m3.metric("Целевое значение", "60%", delta="достигнуто ✓" if target_ok else "не достигнуто")

                    if target_ok:
                        st.success(f"Precision@5 = {mean_p5:.1%} — цель достигнута! ✅")
                    else:
                        st.warning(f"Precision@5 = {mean_p5:.1%} — нужно ≥60%")

                    st.divider()
                    st.markdown("### Детализация по вопросам")

                    # Таблица
                    rows_html = ""
                    for r in sorted(rows, key=lambda x: x["qid"]):
                        s = r["score"]
                        cls = "p-score-high" if s >= 0.8 else ("p-score-mid" if s >= 0.4 else "p-score-low")
                        rows_html += f"""
<div class="precision-row">
  <span class="p-qid">{r['qid']} [{r['difficulty']}, {r['lang']}]</span>
  <span style="color:#718096;font-size:12px;flex:1;padding:0 12px;overflow:hidden">{r['query']}…</span>
  <span class="{cls}">{r['matched']}/{r['total']} → {s:.0%}</span>
</div>"""

                    st.markdown(rows_html, unsafe_allow_html=True)

                    # По сложности
                    st.divider()
                    st.markdown("### По сложности")
                    for diff in ["easy", "medium", "hard"]:
                        diff_scores = [r["score"] for r in rows if r["difficulty"] == diff]
                        if diff_scores:
                            avg = sum(diff_scores) / len(diff_scores)
                            st.progress(avg, text=f"{diff}: {avg:.1%} ({len(diff_scores)} вопросов)")

                    st.info(f"Результаты сохранены в `{results_path}`")

                except Exception as e:
                    st.error(f"Ошибка оценки: {e}")
                    import traceback
                    st.code(traceback.format_exc())
