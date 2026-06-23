import os
import chromadb
from typing import List, Dict, Any
from .schemas.code_chunk import CodeChunk

class ChromaDBClient:
    def __init__(self, db_path: str = "data/vector_db", collection_name: str = "code_base"):
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _make_metadata_safe(self, chunks: List[CodeChunk]) -> List[Dict[str, Any]]:
        """Внутренний вспомогательный метод для очистки метаданных перед сохранением"""
        safe_metadatas = []
        for chunk in chunks:
            chunk_dict = chunk.model_dump()
            safe_metadata = {}
            
            for key, value in chunk_dict.items():
                if value is None:
                    safe_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    safe_metadata[key] = value
                else:
                    safe_metadata[key] = str(value)
                    
            safe_metadatas.append(safe_metadata)
        return safe_metadatas

    def save_chunks(self, chunks: List[CodeChunk], embeddings: List[List[float]]):
        """Пакетное сохранение чанков и их эмбеддингов в дефолтную коллекцию"""
        if not chunks:
            return

        ids = [f"{c.file_path}_{c.element_type}_{c.name}_{c.start_line}" for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = self._make_metadata_safe(chunks)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск ближайших кусков кода по вектору запроса"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted_results = []
        if results and results["documents"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        return formatted_results

    def get_or_create_project_collection(self, project_name: str, dimension: int):
        """
        Нормальное создание коллекции: 
        Привязываем размерность к имени, чтобы при смене моделей ничего не падалo.
        """
        safe_collection_name = f"{project_name}_dim_{dimension}".replace(".", "_").replace("-", "_")
    
        existing_collections = [c.name for c in self.client.list_collections()]
        
        if safe_collection_name in existing_collections:
            return self.client.get_collection(name=safe_collection_name)
        else:
            print(f"🔄 Обнаружена новая модель/размерность ({dimension}). Создаем чистую коллекцию: {safe_collection_name}")
            return self.client.create_collection(
                name=safe_collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def save_chunks_to_collection(self, collection, chunks: List[CodeChunk], embeddings: List[List[float]]):
        """Сохранение чанков с автоматической защитой от дубликатов (UPSERT)"""
        if not chunks:
            return
            
        ids = [f"{c.file_path}_{c.element_type}_{c.name}_{c.start_line}" for c in chunks]
        
        documents = []
        for c in chunks:
            enriched_content = (
                f"Файл: {c.file_path} File path\n"
                f"Сущность: {c.element_type} {c.name}\n"
                f"Код:\n{c.content}"
            )
            documents.append(enriched_content)
        
        metadatas = self._make_metadata_safe(chunks)
        
        collection.upsert(
            ids=ids, 
            embeddings=embeddings, 
            documents=documents, 
            metadatas=metadatas
        )
        print(f"Сохранено {len(chunks)} чанков в ChromaDB")
