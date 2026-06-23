from typing import List, Dict, Any
from ml_core import ChromaDBClient, EmbeddingModelRegistry
from backend.app.services.search_service import SearchService

class RAGController:
    def __init__(self):
        self.db_client = ChromaDBClient()
        self.embedder = EmbeddingModelRegistry()
        self.search_service = SearchService(self.db_client, self.embedder)
        self.project_name = "gymhero"

    def get_indexed_files(self) -> List[str]:
        """Возвращает дерево файлов проекта для левой панели"""
        try:
            current_dimension = self.embedder.get_embedding_dimension()
            collection = self.db_client.get_or_create_project_collection(
                project_name=self.project_name, 
                dimension=current_dimension
            )
            results = collection.get(include=["metadatas"])
            if not results or not results.get("metadatas"):
                return []
                
            unique_files = set()
            for meta in results["metadatas"]:
                if meta and "file_path" in meta:
                    unique_files.add(meta["file_path"])
                    
            return sorted(list(unique_files))
        except Exception as e:
            return []

    def get_available_embedding_models(self) -> List[str]:
        """
        Возвращает доступные RAG-модели эмбеддингов.
        Берем прямо из твоего EmbeddingModelRegistry.
        """
        return EmbeddingModelRegistry.MODELS.keys()

    def switch_embedding_model(self, model_name: str):
        """Переключает активную модель эмбеддингов на лету при изменении в UI"""
        if self.embedder.current_model_key == model_name:
            return
            
        self.embedder.set_active_model(model_name)
        self.search_service = SearchService(self.db_client, self.embedder)
        print(f"Система переключена на поиск через модель: {model_name}")

    def find_relevant_code(self, query: str) -> List[Dict[str, Any]]:
        """
        Ищет релевантный код и возвращает данные в формате твоей схемы CodeChunk
        """
        raw_chunks = self.search_service.search_code(
            project_name=self.project_name, 
            query=query, 
            top_k=5
        )
        
        processed_chunks = []
        for chunk in raw_chunks:
            metadata = chunk.get("metadata", {})
            
            processed_chunks.append({
                "content": chunk.get("content", ""),
                "file_path": metadata.get("file_path", "unknown"),
                "element_type": metadata.get("element_type", "function"),
                "name": metadata.get("name", "unknown"),
                "start_line": int(metadata.get("start_line", 1)),
                "end_line": int(metadata.get("end_line", 1)),
                "docstring": metadata.get("docstring", "")
            })
            
        return processed_chunks
