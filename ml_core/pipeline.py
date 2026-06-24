import os
from typing import List
from .parsers.factory import SplitterFactory
from .schemas.code_chunk import CodeChunk
from .db_client import ChromaDBClient
from .models import EmbeddingModelRegistry

class IndexingPipeline:
    def __init__(self, db_client: ChromaDBClient, embedder: EmbeddingModelRegistry):
        self.db = db_client
        self.embedder = embedder
        self.supported_ext = SplitterFactory.get_supported_extensions()


    def index_project(self, project_path: str) -> dict:
        """
        Полный пайплайн индексации директории.
        Возвращает статистику для вывода метрик на фронтенд.
        """
        if not os.path.exists(project_path):
            raise FileNotFoundError(f"Путь {project_path} не существует.")

        project_name = os.path.basename(os.path.normpath(project_path))
        current_dimension = self.embedder.get_embedding_dimension() 

        collection = self.db.get_or_create_project_collection(
            project_name=project_name, 
            dimension=current_dimension
        )

        all_chunks: List[CodeChunk] = []
        stats = {"total_files": 0, "processed_files": 0, "chunks_count": 0, "languages": {}}

        print(f"Старт индексации проекта: {project_name}")

        for root, _, files in os.walk(project_path):
            for file in files:
                stats["total_files"] += 1
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                
                if ext in self.supported_ext:
                    stats["processed_files"] += 1
                    file_path = os.path.join(root, file)
                    
                    try:
                        parser = SplitterFactory.get_parser(file_path)
                        file_chunks = parser.parse_file(file_path)
                        
                        for chunk in file_chunks:
                            clean_path = chunk.file_path.replace("\\", "/")
                            chunk.file_path = clean_path
                        
                        all_chunks.extend(file_chunks)
                        
                        lang = ext.replace('.', '')
                        stats["languages"][lang] = stats["languages"].get(lang, 0) + len(file_chunks)
                    except Exception as e:
                        print(f"Ошибка при парсинге файла {file_path}: {e}")

        stats["chunks_count"] = len(all_chunks)

        if all_chunks:
            print(f"Генерация эмбеддингов для {len(all_chunks)} фрагментов кода...")
    
            texts_to_embed = [chunk.content for chunk in all_chunks]
            embeddings = self.embedder.embed_texts(texts_to_embed)
            print("Сохранение векторов в ChromaDB...")
            self.db.save_chunks_to_collection(collection, all_chunks, embeddings)
            
        print(f"Индексация проекта '{project_name}' успешно завершена!")
        return stats
