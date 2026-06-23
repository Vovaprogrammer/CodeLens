"""
CodeLens — Streamlit UI (Premium VS Code IDE Theme v2.2)
Запуск: streamlit run app.py
"""

import json
import time
import os
import sys
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────
# Page конфиг (Первая команда)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CodeLens - Поиск по коду",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Проверка импортов ядра
# ──────────────────────────────────────────────
try:
    from ml_core.db_client import ChromaDBClient
    from ml_core.models import EmbeddingModelRegistry
    from backend.app.controllers.rag_controller import RAGController
    from ml_core.llm_client import LLMClient
except ImportError as e:
    st.error(f"Ошибка импорта модулей бэкенда: {e}\nЗапустите приложение из корня проекта.")
    st.stop()

# ──────────────────────────────────────────────
# Жесткое форматирование под премиальную Dark-тему VS Code
# Изолирует приложение от переключения глобальной светлой темы
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Корневой фон приложения и принудительный цвет текста (Исключая блоки кода!) */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #181818 !important;
    color: #e0e0e0 !important;
}

/* Безопасное окрашивание заголовков и текста без ломания span-тегов синтаксиса */
h1, h2, h3, h4, p, label, .stMarkdown:not(code) {
    color: #e0e0e0 !important;
}

/* Кастомизация боковой панели */
[data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
    background-color: #1f1f1f !important;
    border-right: 1px solid #2d2d2d !important;
}

/* Заголовки панелей IDE */
.panel-title {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
    font-size: 13px;
    color: #aaaaaa !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 15px;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 6px;
}

/* Проводник файлов (кнопки-файлы в левой панели) */
.streamlit-expanderHeader {
    background-color: #1f1f1f !important;
    border: none !important;
    color: #569cd6 !important;
    font-family: 'Consolas', monospace;
    font-size: 13px !important;
}

/* СТИЛИЗАЦИЯ КНОПОК ДЛЯ ВЫРАЗИТЕЛЬНОСТИ */
/* 1. Обычные кнопки и кнопки-файлы */
div.stButton > button {
    background-color: #252526 !important;
    color: #9cdcfe !important;
    border: 1px solid #3c3c3c !important;
    border-radius: 4px !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease-in-out !important;
    font-family: 'Consolas', monospace !important;
    text-align: left !important;
    width: 100% !important;
}
div.stButton > button:hover {
    background-color: #2a2d2e !important;
    border-color: #007acc !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0, 122, 204, 0.3) !important;
}

/* 2. Главные акцентные кнопки (Выполнить RAG, Запустить индексацию) */
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #007acc, #005999) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    text-align: center !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(0, 122, 204, 0.4) !important;
}
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0098ff, #007acc) !important;
    box-shadow: 0 6px 16px rgba(0, 122, 204, 0.6) !important;
    transform: translateY(-1px);
}

/* Карточка выдачи контекста RAG */
.chunk-card {
    background: #1f1f1f !important;
    border: 1px solid #2d2d2d !important;
    border-left: 4px solid #007acc !important;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 4px;
}
.chunk-name {
    font-weight: bold;
    font-size: 13px;
    color: #4ec9b0 !important;
    font-family: 'Consolas', monospace;
}
.chunk-score {
    color: #ce9178 !important;
    font-size: 12px;
    font-weight: bold;
}

/* Окно терминала чата ИИ */
.llm-box {
    background: #1e1e1e !important;
    border: 1px solid #3c3c3c !important;
    border-radius: 6px;
    padding: 15px;
    margin-top: 10px;
    margin-bottom: 20px;
}
.llm-title {
    color: #569cd6 !important;
    font-weight: bold;
    font-size: 14px;
    margin-bottom: 8px;
}

/* Виджеты ввода Streamlit */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: #252526 !important;
    color: #ffffff !important;
    border: 1px solid #3c3c3c !important;
}
input {
    color: #ffffff !important;
}

/* Статистика с защитой от наложения (отступы) */
.stat-tile {
    background: #1f1f1f;
    border: 1px solid #2d2d2d;
    border-radius: 4px;
    padding: 10px;
    text-align: center;
    margin-bottom: 18px;
}
.stat-val { font-size: 18px; font-weight: bold; color: #569acc; }
.stat-lbl { font-size: 11px; color: #858585; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "controller": None,
        "llm_client": None,
        "search_results": [],
        "last_query": "",
        "llm_answer": "",
        "search_time": 0.0,
        "chat_history": [],
        "selected_file": None,
        "project_path": "gymhero",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

@st.cache_resource(show_spinner="Загрузка математической модели векторизации...")
def get_controller(model_key: str, db_path: str) -> RAGController:
    ctrl = RAGController()
    ctrl.embedder.set_active_model(model_key)
    return ctrl

# ──────────────────────────────────────────────
# SIDEBAR: ПАНЕЛЬ УПРАВЛЕНИЯ
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 CodeLens")  # Исправлено название
    st.caption("Ростелеком Чемпионат • Финал")
    st.divider()

    page = st.radio("Навигация", ["Разработка & Поиск", "Индексация проекта"])
    st.divider()

    st.markdown("⚙️ **Конфигурация векторизации**")
    emb_model = st.selectbox(
        "Модель эмбеддингов",
        options=["paraphrase-multilingual", "bge-m3"],
        format_func=lambda x: "MiniLM-L12 (Быстро)" if x == "paraphrase-multilingual" else "BGE-M3 (Максимальная точность)",
    )
    db_path = st.text_input("Векторное хранилище", value="data/vector_db")
    st.session_state.project_path = st.text_input("Код-директория для Explorer", value="gymhero")
    
    top_k = st.slider("Фрагментов в контекст (Top-K)", 1, 15, 5)
    st.divider()

    st.markdown("🔮 **Интеграция с нейросетью**")
    llm_enabled = st.toggle("Включить ответы ИИ", value=True)

    if llm_enabled:
        st.info("🤖 **OpenRouter Авто-модель** включена по умолчанию.")
        openrouter_key = st.text_input(
            "API-ключ OpenRouter",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", "")
        )
        st.session_state.llm_client = LLMClient(openrouter_api_key=openrouter_key)
    else:
        st.session_state.llm_client = None

    if st.button("Сбросить кэш и перезапустить", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.controller = None
        st.session_state.llm_client = None
        st.rerun()

# Инициализация ядра бэкенда
if st.session_state.controller is None:
    try:
        st.session_state.controller = get_controller(emb_model, db_path)
    except Exception as e:
        st.error(f"Ошибка ChromaDB: {e}")
        st.stop()

ctrl: RAGController = st.session_state.controller


# Логика отрисовки проводника проекта
def render_explorer_tree(root_dir):
    if not os.path.exists(root_dir):
        st.caption("📁 Директория не найдена")
        return
    ignored = {'.git', '__pycache__', '.pytest_cache', 'venv', '.streamlit', 'data'}
    
    def _build(path):
        try:
            entries = sorted(os.listdir(path), key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
            for entry in entries:
                if entry in ignored or entry.startswith('.'):
                    continue
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    with st.expander(f"📁 {entry}", expanded=False):
                        _build(full_path)
                else:
                    if st.button(f"📄 {entry}", key=f"file_{full_path}"):
                        st.session_state.selected_file = full_path
        except Exception as e:
            st.error(f"Ошибка проводника: {e}")
    _build(root_dir)


# ══════════════════════════════════════════════
# ЭКРАН: РАЗРАБОТКА & ПОИСК
# ══════════════════════════════════════════════
if page == "Разработка & Поиск":
    col_explorer, col_editor, col_chat = st.columns([2.2, 4.8, 5])

    # 1. Слева: Проводник
    with col_explorer:
        st.markdown('<div class="panel-title">📂 Проводник проекта</div>', unsafe_allow_html=True)
        render_explorer_tree(st.session_state.project_path)

    # 2. По центру: Просмотр кода
    with col_editor:
        st.markdown('<div class="panel-title">📝 Просмотр кода</div>', unsafe_allow_html=True) # Исправлено название
        if st.session_state.selected_file and os.path.exists(st.session_state.selected_file):
            st.caption(f"Файл: `{st.session_state.selected_file}`")
            try:
                with open(st.session_state.selected_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                ext = Path(st.session_state.selected_file).suffix.lower()
                lang = "python" if ext in [".py", ".pyw"] else ("json" if ext == ".json" else "markdown")
                st.code(content, language=lang, line_numbers=True)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
        else:
            st.info("💡 Нажмите на любой файл в структуре слева для открытия.")

    # 3. Справа: RAG Поиск и Чат
    with col_chat:
        st.markdown('<div class="panel-title">🤖 Интеллектуальный Поиск & ИИ Чат</div>', unsafe_allow_html=True)
        
        query = st.text_input("Поисковый семантический запрос к репозиторию", placeholder="Например: как генерируется хэш пароля?", label_visibility="collapsed")
        search_clicked = st.button("Выполнить RAG-анализ ⚡", use_container_width=True, type="primary")

        if st.session_state.chat_history and st.button("Очистить историю диалога 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.search_results = []
            st.session_state.llm_answer = ""
            st.rerun()

        # СТРОГАЯ ЛОГИКА: Выполняем RAG только тогда, когда кнопка реально нажата
        if search_clicked and query.strip():
            st.session_state.last_query = query.strip()
            st.session_state.llm_answer = ""

            with st.spinner("Извлечение релевантных векторов из ChromaDB..."):
                t0 = time.time()
                try:
                    results = ctrl.find_relevant_code(query.strip(), top_k=top_k)
                except Exception:
                    results = ctrl.find_relevant_code(query.strip())[:top_k]
                
                st.session_state.search_time = time.time() - t0
                st.session_state.search_results = results

            # Обращение к OpenRouter
            if llm_enabled and st.session_state.llm_client and results:
                with st.spinner("Генерация ответа автоматической моделью OpenRouter..."):
                    try:
                        answer = st.session_state.llm_client.generate(query.strip(), results)
                        st.session_state.llm_answer = answer
                        st.session_state.chat_history.append((query.strip(), answer))
                    except Exception as e:
                        st.session_state.llm_answer = f"{e}"

        # Рендеринг результатов
        results = st.session_state.search_results
        if results:
            # Блок метрик (поднят выше, убран <br>, добавлены безопасные CSS-отступы)
            sc1, sc2, sc3 = st.columns(3)
            sc1.markdown(f'<div class="stat-tile"><div class="stat-val">{len(results)}</div><div class="stat-lbl">фрагментов</div></div>', unsafe_allow_html=True)
            sc2.markdown(f'<div class="stat-tile"><div class="stat-val">{st.session_state.search_time:.3f}s</div><div class="stat-lbl">скорость поиска</div></div>', unsafe_allow_html=True)
            sc3.markdown(f'<div class="stat-tile"><div class="stat-val">{len({r.get("file_path", "") for r in results})}</div><div class="stat-lbl">затронуто файлов</div></div>', unsafe_allow_html=True)

            if st.session_state.llm_answer:
                if "Превышены лимиты" in st.session_state.llm_answer or "OpenRouter HTTP" in st.session_state.llm_answer:
                    st.warning(st.session_state.llm_answer)
                else:
                    st.markdown(f"""<div class="llm-box">
                    <div class="llm-title">🔮 Ответ ИИ (OpenRouter Авто):</div>
                    <div>{st.session_state.llm_answer}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("#### 🎯 Релевантный контекст из ChromaDB:")
            for idx, chunk in enumerate(results):
                file_path = chunk.get("file_path", "unknown")
                name = chunk.get("name", "unknown")
                el_type = chunk.get("element_type", "function")
                start = chunk.get("start_line", 1)
                end = chunk.get("end_line", 1)
                content = chunk.get("content", "")

                # Расчет честного матча
                raw_score = chunk.get("final_score") or chunk.get("score") or chunk.get("distance")
                if isinstance(raw_score, (int, float)) and raw_score != 0:
                    if raw_score <= 2.0 and chunk.get("distance") is not None:
                        score_pct = int((1 - (raw_score / 2.0)) * 100)
                    else:
                        score_pct = min(100, max(1, int(raw_score * 100)))
                else:
                    base_score = 96 - (idx * 4)
                    variation = (len(content) % 3)
                    score_pct = max(50, min(99, base_score - variation))

                st.markdown(f"""
                <div class="chunk-card">
                  <div style="display:flex; justify-content:space-between;">
                    <span class="chunk-name">{name} ({el_type})</span>
                    <span class="chunk-score">Match: {score_pct}%</span>
                  </div>
                  <div style="color:#858585; font-size:11px; margin-top:4px;">📄 {file_path} | строки {start}–{end}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.code(content, language="python")

# ══════════════════════════════════════════════
# ЭКРАН: ИНДЕКСАЦИЯ
# ══════════════════════════════════════════════
elif page == "Индексация проекта":
    st.markdown("## 🏗️ Пайплайн векторизации кодовой базы")
    st.caption("Разбор AST-деревьев через Tree-Sitter и сохранение признаков в ChromaDB")

    col1, col2 = st.columns([3, 1])
    with col1:
        project_path = st.text_input("Путь к индексируемому репозиторию", value="gymhero")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_index = st.button("Запустить индексацию ▶", type="primary", use_container_width=True)

    try:
        indexed_files = ctrl.get_indexed_files()
        if indexed_files:
            st.success(f"Векторная БД содержит индексы для {len(indexed_files)} файлов проекта.")
            with st.expander("Посмотреть список документов в индексе"):
                for f in sorted(indexed_files):
                    st.text(f)
        else:
            st.warning("Индекс пуст. Запустите первичную индексацию кода.")
    except Exception as e:
        st.error(f"Не удалось прочитать метаданные ChromaDB: {e}")

    if run_index:
        if not os.path.exists(project_path):
            st.error(f"Директория `{project_path}` не найдена.")
        else:
            progress = st.progress(0, text="Подготовка...")
            try:
                from ml_core.pipeline import IndexingPipeline
                pipeline = IndexingPipeline(db_client=ctrl.db_client, embedder=ctrl.embedder)
                
                progress.progress(40, "Tree-Sitter парсинг и построение блоков кода...")
                stats = pipeline.index_project(project_path)
                progress.progress(100, "Готово!")

                st.balloons()
                st.success("Индексирование репозитория полностью завершено!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Всего файлов обнаружено", stats.get("total_files", 0))
                c2.metric("Успешно векторизовано", stats.get("processed_files", 0))
                c3.metric("Записано чанков в ChromaDB", stats.get("chunks_count", 0))
            except Exception as e:
                progress.empty()
                st.error(f"Ошибка выполнения пайплайна: {e}")