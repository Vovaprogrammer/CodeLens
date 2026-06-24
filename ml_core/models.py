import gc
import torch
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModelRegistry:
    MODELS = {
        "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "bge-m3": "BAAI/bge-m3"                                    
    }

    def __init__(self, model_key: str = "paraphrase-multilingual"):
        if model_key not in self.MODELS:
            raise ValueError(f"Модель {model_key} не поддерживается. Выберите из: {list(self.MODELS.keys())}")
        
        self.current_model_key = None
        
        self.set_active_model(model_key)

    def set_active_model(self, model_key: str):
        """
        Переключает активную модель.
        Используется контроллером RAGController при выборе модели в UI Streamlit.
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Модель {model_key} не поддерживается.")
            
        if self.current_model_key == model_key:
            return

        if self.current_model_key is not None:
            print(f"Выгрузка и полное удаление модели {self.current_model_key}...")
            
            self.model = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Память от {self.current_model_key} полностью освобождена.")
        self.current_model_key = model_key
        self.model_name = self.MODELS[model_key]
        
        print(f"Загрузка модели эмбеддингов: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print(f"Модель {model_key} успешно инициализирована.")

    def get_embedding_dimension(self) -> int:
        """Возвращает размерность вектора модели (нужно для инициализации коллекции ChromaDB)"""
        return self.model.get_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Генерация эмбеддингов для списка текстов (батчем)"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=8)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Генерация эмбеддинга для одного поискового запроса"""
        return self.model.encode(query).tolist()
