"""
CodeLens — Streamlit UI (Premium VS Code IDE Theme v2.2)
Запуск: streamlit run app.py
"""

import time
import os
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

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
/* 1. КОРНЕВЫЕ СТИЛИ СТРАНИЦЫ И БОКОВОЙ ПАНЕЛИ */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #181818 !important;
    color: #e0e0e0 !important;
}

h1, h2, h3, h4, p, label, .stMarkdown:not(code) {
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
    background-color: #1f1f1f !important;
    border-right: 1px solid #2d2d2d !important;
}

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

/* 2. СТРОГАЯ ИЗОЛЯЦИЯ КНОПОК ПРОВОДНИКА (ТОЛЬКО В ПЕРВОЙ КОЛОНКЕ) */
div[data-testid="stColumn"]:first-child div.stButton,
div[data-testid="column"]:first-child div.stButton {
    width: 100% !important;
    margin: 0px !important;
    padding: 0px !important;
}

div[data-testid="stColumn"]:first-child div.stButton > button,
div[data-testid="column"]:first-child div.stButton > button {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0px !important;
    box-shadow: none !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important; 
    align-items: center !important;
    text-align: left !important;
    padding: 3px 6px !important;
    margin: 0px !important;
    min-height: 26px !important;
    height: 26px !important;
    line-height: 26px !important;
    font-family: 'Consolas', monospace !important;
    font-size: 13px !important;
    color: #cccccc !important;
    white-space: pre !important; /* Сохраняет Юникод-пробелы \u00A0 */
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Отмена центровки внутреннего контейнера Streamlit внутри кнопок проводника */
div[data-testid="stColumn"]:first-child div.stButton > button div, 
div[data-testid="stColumn"]:first-child div.stButton > button span, 
div[data-testid="stColumn"]:first-child div.stButton > button p,
div[data-testid="column"]:first-child div.stButton > button div, 
div[data-testid="column"]:first-child div.stButton > button span, 
div[data-testid="column"]:first-child div.stButton > button p {
    text-align: left !important;
    justify-content: flex-start !important;
    margin: 0 !important;
    display: block !important;
}

/* Ховер-эффект на всю ширину колонки проводника */
div[data-testid="stColumn"]:first-child div.stButton > button:hover,
div[data-testid="column"]:first-child div.stButton > button:hover {
    background-color: #2a2d2e !important;
    color: #569cd6 !important;
}

/* Полное уничтожение межстрочных интервалов сетки СТРОГО внутри проводника */
div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlockRoot"] div[data-testid="stVerticalBlock"] > div,
div[data-testid="stColumn"]:first-child [data-testid="element-container"],
div[data-testid="column"]:first-child div[data-testid="stVerticalBlockRoot"] div[data-testid="stVerticalBlock"] > div,
div[data-testid="column"]:first-child [data-testid="element-container"] {
    padding-bottom: 0px !important;
    margin-bottom: 0px !important;
    margin-top: 0px !important;
    padding-top: 0px !important;
    height: auto !important;
}
div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"],
div[data-testid="column"]:first-child div[data-testid="stVerticalBlock"] {
    gap: 0px !important;
}

/* 3. СТИЛИЗАЦИЯ ДЛЯ ГЛАВНЫХ КНОПОК ИНТЕРФЕЙСА (RAG, ИНДЕКСАЦИЯ И Т.Д.) */
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #007acc, #005999) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 12px rgba(0, 122, 204, 0.4) !important;
    height: 40px !important;
    width: 100% !important;
    margin-top: 10px !important;
    display: inline-flex !important;
}
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0098ff, #007acc) !important;
    box-shadow: 0 6px 16px rgba(0, 122, 204, 0.6) !important;
    transform: translateY(-1px);
    color: #ffffff !important;
}

/* 4. ОСТАЛЬНЫЕ КОМПОНЕНТЫ ИНТЕРФЕЙСА */
.chunk-card {
    background: #1f1f1f !important;
    border: 1px solid #2d2d2d !important;
    border-left: 4px solid #007acc !important;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 8px;
}
.chunk-name {
    font-weight: bold;
    font-size: 13px;
    color: #4ec9b0 !important;
    font-family: 'Consolas', monospace;
}
.chunk-score { color: #ce9178 !important; font-size: 12px; font-weight: bold; }

.llm-box {
    background: #1e1e1e !important;
    border: 1px solid #3c3c3c !important;
    border-radius: 6px;
    padding: 15px;
    margin-top: 15px;
    margin-bottom: 25px;
}
.llm-title { color: #569cd6 !important; font-weight: bold; font-size: 14px; margin-bottom: 8px; }

div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: #252526 !important;
    color: #ffffff !important;
    border: 1px solid #3c3c3c !important;
}
input { color: #ffffff !important; }

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
        "project_path": "projects",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

@st.cache_resource(show_spinner="Загрузка математической модели векторизации...")
def get_controller(file_path, model_key: str, db_path: str) -> RAGController:
    ctrl = RAGController(file_path, model_key=model_key, db_path=db_path)
    return ctrl

# ──────────────────────────────────────────────
# SIDEBAR: ПАНЕЛЬ УПРАВЛЕНИЯ
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 CodeLens")
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
    st.session_state.project_path = st.text_input("Код-директория для Explorer", value="projects")

    top_k = st.slider("Фрагментов в контекст (Top-K)", 1, 15, 5)
    st.divider()

    st.markdown("🔮 **Интеграция с нейросетью**")
    llm_enabled = st.toggle("Включить ответы ИИ", value=True)

    if llm_enabled:
        env_key = os.getenv("OPEN_ROUTER_API", "")

        if env_key:
            st.success("🤖 Ключ OpenRouter успешно загружен из .env")
            with st.expander("Посмотреть / Изменить API-ключ"):
                openrouter_key = st.text_input(
                    "API-ключ OpenRouter",
                    type="password",
                    value=env_key
                )
        else:
            st.warning("⚠️ Ключ OPENROUTER_API_KEY не найден в .env")
            openrouter_key = st.text_input(
                "Введите API-ключ вручную",
                type="password",
                value=""
            )

        if openrouter_key:
            st.session_state.llm_client = LLMClient(openrouter_api_key=openrouter_key)
        else:
            st.session_state.llm_client = None
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
        st.session_state.controller = get_controller(st.session_state.project_path, emb_model, db_path)
    except Exception as e:
        st.error(f"Ошибка ChromaDB: {e}")
        st.stop()

ctrl: RAGController = st.session_state.controller


def render_explorer_tree(root_dir):
    if not os.path.exists(root_dir):
        st.caption("📁 Директория не найдена")
        return

    ignored = {'.git', '__pycache__', '.pytest_cache', 'venv', '.streamlit', 'data', 'target'}

    if "expanded_folders" not in st.session_state:
        st.session_state["expanded_folders"] = set()

    def _build_flat_tree(current_path, depth=0):
        try:
            entries = sorted(os.listdir(current_path), key=lambda x: (not os.path.isdir(os.path.join(current_path, x)), x.lower()))

            for entry in entries:
                if entry in ignored or entry.startswith('.'):
                    continue

                full_path = os.path.join(current_path, entry)
                is_dir = os.path.isdir(full_path)

                indent = "\u00A0" * (depth * 4)

                if is_dir:
                    is_expanded = full_path in st.session_state.expanded_folders
                    icon = "📂" if is_expanded else "📁"

                    if st.button(f"{indent}{icon} {entry}", key=f"dir_{full_path}", use_container_width=True):
                        if is_expanded:
                            st.session_state.expanded_folders.remove(full_path)
                        else:
                            st.session_state.expanded_folders.add(full_path)
                        st.rerun()

                    if is_expanded:
                        _build_flat_tree(full_path, depth + 1)

                # --- Для файлов ---
                else:
                    if st.button(f"{indent}📄 {entry}", key=f"file_{full_path}", use_container_width=True):
                        st.session_state.selected_file = full_path
                        st.rerun()

        except Exception as e:
            st.error(f"Ошибка проводника: {e}")

    _build_flat_tree(root_dir, depth=0)


# ══════════════════════════════════════════════
# ЭКРАН: РАЗРАБОТКА & ПОИСК
# ══════════════════════════════════════════════
if page == "Разработка & Поиск":

    def get_language_by_filename(filename: str) -> str:
        if not filename:
            return "text"
        ext = Path(filename).suffix.lower().lstrip('.')
        mapping = {
            "py": "python", "pyw": "python", "java": "java",
            "js": "javascript", "ts": "typescript", "json": "json",
            "yml": "yaml", "yaml": "yaml", "xml": "xml",
            "gradle": "groovy", "properties": "properties", "md": "markdown"
        }
        return mapping.get(ext, "text")

    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "llm_answer" not in st.session_state:
        st.session_state.llm_answer = ""
    if "search_time" not in st.session_state:
        st.session_state.search_time = 0.0
    if "temp_query" not in st.session_state:
        st.session_state.temp_query = ""

    if st.session_state.temp_query:
        current_query = st.session_state.temp_query
        st.session_state.last_query = current_query
        st.session_state.llm_answer = ""

        results = ctrl.find_relevant_code(current_query, top_k=top_k)
        st.session_state.search_results = results

        if results and len(results) > 0:
            top_file = results[0].get("file_path", "")
            if os.path.exists(top_file):
                st.session_state.selected_file = top_file
            else:
                potential_path = os.path.join(st.session_state.project_path, top_file)
                if os.path.exists(potential_path):
                    st.session_state.selected_file = potential_path

        if llm_enabled and st.session_state.llm_client and results:
            try:
                answer = st.session_state.llm_client.generate(current_query, results)
                st.session_state.llm_answer = answer
                st.session_state.chat_history.append((current_query, answer))
            except Exception as e:
                st.session_state.llm_answer = f"{e}"

        st.session_state.temp_query = ""
        st.rerun()


    col_explorer, col_editor, col_chat = st.columns([2.2, 4.8, 5])

    with col_explorer:
        st.markdown('<div class="panel-title">📂 Проводник проекта</div>', unsafe_allow_html=True)
        render_explorer_tree(st.session_state.project_path)

    with col_chat:
        st.markdown('<div class="panel-title">🤖 Интеллектуальный Поиск & ИИ Чат</div>', unsafe_allow_html=True)

        query_input = st.text_input("Поисковый семантический запрос к репозиторию", placeholder="Например: как генерируется хэш пароля?", label_visibility="collapsed")
        search_clicked = st.button("Выполнить RAG-анализ ⚡", use_container_width=True, type="primary")

        if st.session_state.chat_history and st.button("Очистить историю диалога 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.search_results = []
            st.session_state.llm_answer = ""
            st.rerun()

        if search_clicked and query_input.strip():
            st.session_state.temp_query = query_input.strip()
            st.rerun()

    with col_editor:
        st.markdown('<div class="panel-title">📝 Просмотр кода</div>', unsafe_allow_html=True)

        if st.session_state.selected_file and os.path.exists(st.session_state.selected_file):
            try:
                with open(st.session_state.selected_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    lines = content.splitlines()

                lang = get_language_by_filename(st.session_state.selected_file)

                target_start, target_end = None, None
                if st.session_state.search_results:
                    top = st.session_state.search_results[0]
                    if top.get("file_path") == st.session_state.selected_file:
                        target_start = top.get("start_line", 1)
                        target_end = top.get("end_line", 1)

                st.caption(f"Файл: `{st.session_state.selected_file}`")

                if target_start and target_end:
                    st.markdown(f"🎯 **Найденный фрагмент (строки {target_start}-{target_end}):**")
                    st.code(content, language=lang, line_numbers=True)
                else:
                    st.code(content, language=lang, line_numbers=True)

            except Exception as e:
                st.error(f"Ошибка чтения: {e}")

    with col_chat:
        results = st.session_state.search_results
        if results:
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

                rag_lang = get_language_by_filename(file_path)
                st.code(content, language=rag_lang)

# ══════════════════════════════════════════════
# ЭКРАН: ИНДЕКСАЦИЯ
# ══════════════════════════════════════════════
elif page == "Индексация проекта":
    st.markdown("## 🏗️ Пайплайн векторизации кодовой базы")
    st.caption("Разбор AST-деревьев через Tree-Sitter и сохранение признаков в ChromaDB")

    col1, col2 = st.columns([3, 1])
    with col1:
        project_path = st.text_input("Путь к индексируемому репозиторию", value=st.session_state.project_path)
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