import os
import json
from .search_service import SearchService
from ml_core import ChromaDBClient, EmbeddingModelRegistry
from ml_core.pipeline import IndexingPipeline 

import re

def fix_chunk_id_format(chunk_id: str, project_name: str) -> str:
    """
    Превращает кривой формат из ChromaDB в формат для score.py
    Пример:
    'gymhero/crud/user.py_function_is_super_user_25' -> 'gymhero/crud/user.py:UserCRUDRepository.is_super_user:25'
    """
    double_prefix = f"{project_name}/{project_name}/"
    if chunk_id.startswith(double_prefix):
        chunk_id = chunk_id.replace(double_prefix, f"{project_name}/", 1)
    match_line = re.search(r'_(\d+)$', chunk_id)
    if not match_line:
        return chunk_id
    
    lineno = match_line.group(1)
    id_without_line = chunk_id[:match_line.start()]
    if '_function_' in id_without_line:
        path, name = id_without_line.split('_function_', 1)
    elif '_class_' in id_without_line:
        path, name = id_without_line.split('_class_', 1)
    else:
        return chunk_id

    return f"{path}:{name}:{lineno}"


def generate_submission(
    project_name: str,
    questions_json_path: str,
    output_submission_path: str,
    search_service: SearchService
):
    if not os.path.exists(questions_json_path):
        print(f"❌ Ошибка: Файл с вопросами {questions_json_path} не найден.")
        return

    print(f"Загрузка вопросов из {questions_json_path}...")
    with open(questions_json_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    submission_results = []
    total_questions = len(questions_data)

    print(f"Начало обработки {total_questions} вопросов для формирования сабмишна...")

    for idx, item in enumerate(questions_data):
        question_id = item.get("question_id") or item.get("id") or f"q_{idx+1:02d}"
        query = item.get("query") or item.get("question")

        if not query:
            print(f"⚠️ Пропущен вопрос {idx+1}: отсутствует текст запроса.")
            continue

        try:
            search_results = search_service.search_code(
                project_name=project_name, 
                query=query, 
                top_k=5
            )

            top_5_chunks = []
            for res in search_results:
                raw_id = res["id"]
                clean_id = fix_chunk_id_format(raw_id, project_name)
                top_5_chunks.append(clean_id)

            if len(top_5_chunks) == 0:
                print(f"   ⚠️ Внимание: Для '{question_id}' база вернула 0 результатов!")

            submission_results.append({
                "question_id": question_id,
                "top_5_chunks": top_5_chunks
            })

            print(f"   [{idx+1}/{total_questions}] Обработан {question_id}. Чанков: {len(top_5_chunks)}")

        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса {question_id}: {e}")
            submission_results.append({
                "question_id": question_id,
                "top_5_chunks": []
            })

    print(f"Сохранение сабмишна в {output_submission_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_submission_path)), exist_ok=True)
    
    with open(output_submission_path, "w", encoding="utf-8") as f:
        json.dump(submission_results, f, ensure_ascii=False, indent=2)

    print("🎉 Сабмишн успешно сгенерирован и адаптирован под score.py!")


if __name__ == "__main__":
    print("Инициализация клиентов для генерации...")
    db_client = ChromaDBClient()
    embedder = EmbeddingModelRegistry()
    
    PROJECT_SOURCE_PATH = "gymhero" 
    PROJECT_NAME = "gymhero" 
    QUESTIONS_PATH = "eval_questions.json"  
    OUTPUT_PATH = "results.json"        

    print("\n--- Шаг 1: Проверка и индексация исходного кода ---")
    if os.path.exists(PROJECT_SOURCE_PATH):
        pipeline = IndexingPipeline(db_client=db_client, embedder=embedder)
        indexing_stats = pipeline.index_project(PROJECT_SOURCE_PATH)
        print(f"📊 Статистика индексации: {indexing_stats}")
    else:
        print(f"⚠️ Путь '{PROJECT_SOURCE_PATH}' не найден. Работаем с существующей базой.")
    
    print("\n--- Шаг 2: Генерация сабмишна ---")
    search_service = SearchService(db_client=db_client, embedder=embedder)
    generate_submission(PROJECT_NAME, QUESTIONS_PATH, OUTPUT_PATH, search_service)
