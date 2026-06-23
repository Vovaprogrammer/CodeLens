"""
CodeLens — индексирующий скрипт.
Использование:
    python index.py <путь_к_директории> [--model paraphrase-multilingual|bge-m3]
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="CodeLens: индексация кодовой базы в ChromaDB"
    )
    parser.add_argument("project_path", help="Путь к директории с кодом")
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual",
        choices=["paraphrase-multilingual", "bge-m3"],
        help="Модель эмбеддингов (default: paraphrase-multilingual)",
    )
    parser.add_argument(
        "--db-path",
        default="data/vector_db",
        help="Путь к хранилищу ChromaDB (default: data/vector_db)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.project_path):
        print(f"[ОШИБКА] Путь не найден: {args.project_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CodeLens Indexer")
    print(f"  Проект   : {args.project_path}")
    print(f"  Модель   : {args.model}")
    print(f"  БД       : {args.db_path}")
    print()

    from ml_core.db_client import ChromaDBClient
    from ml_core.models import EmbeddingModelRegistry
    from ml_core.pipeline import IndexingPipeline

    db_client = ChromaDBClient(db_path=args.db_path)
    embedder = EmbeddingModelRegistry(model_key=args.model)
    pipeline = IndexingPipeline(db_client=db_client, embedder=embedder)

    stats = pipeline.index_project(args.project_path)

    print()
    print("Индексация завершена!")
    print(f"  Всего файлов    : {stats.get('total_files', 0)}")
    print(f"  Обработано      : {stats.get('processed_files', 0)}")
    print(f"  Чанков создано  : {stats.get('chunks_count', 0)}")
    langs = stats.get("languages", {})
    if langs:
        print(f"  Языки           : {', '.join(f'{k}={v}' for k, v in langs.items())}")
    print()
    print("Запусти интерфейс: streamlit run app.py")


if __name__ == "__main__":
    main()
