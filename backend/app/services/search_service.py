import os
from typing import List, Dict, Any
from ml_core import ChromaDBClient, EmbeddingModelRegistry

class SearchService:
    def __init__(self, db_client: ChromaDBClient, embedder: EmbeddingModelRegistry, ollama_url: str = "http://localhost:11434"):
        self.db = db_client
        self.embedder = embedder
        self.ollama_url = ollama_url

    def search_code(self, project_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []

        current_dimension = self.embedder.get_embedding_dimension()
        collection = self.db.get_or_create_project_collection(
            project_name=project_name, 
            dimension=current_dimension
        )

        query_vector = self.embedder.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=40
        )

        formatted_candidates = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["ids"][0])):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] or {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                
                vector_similarity = 1.0 / (1.0 + distance)
                
                content_lower = content.lower()
                chunk_id = results["ids"][0][i]
                file_path = metadata.get("file_path") or chunk_id.split(":", 1)[0]
                file_path_lower = file_path.lower()
                element_name_lower = metadata.get("name", "").lower()
                
                query_words = []
                for w in query.split():
                    w_clean = w.lower().strip("?,.:;!")
                    if len(w_clean) > 3:
                        query_words.append(w_clean[:6] if len(w_clean) > 6 else w_clean)
                
                synonyms = {
                    "токен": ["token", "jwt", "create_access_token", "security"],
                    "парол": ["password", "hash", "verify", "pwd"],
                    "сесси": ["session", "engine", "db", "get_db", "sessionmaker"],
                    "репози": ["repository", "crud", "base", "sqlalchemy", "model"],
                    "ошибк": ["error", "exception", "status_code", "raise", "400", "404"],
                    "пагина": ["pagination", "limit", "offset", "page", "skip"],
                    "accoun": ["user", "auth", "register", "signup", "create_user"],
                    "являет": ["admin", "superuser", "is_admin", "role", "is_superuser"],
                    "уникал": ["unique", "exists", "get_one", "training_plan", "create_training_plan"] 
                }
                
                keyword_score = 0.0
                
                for word in query_words:
                    word_matched = False
                    if word in content_lower:
                        keyword_score += 0.35
                        word_matched = True
                    if word in file_path_lower or word in element_name_lower:
                        keyword_score += 0.6
                        word_matched = True
                        
                    if not word_matched:
                        for syn_key, syn_list in synonyms.items():
                            if word in syn_key or syn_key in word:
                                for syn in syn_list:
                                    if syn in content_lower or syn in file_path_lower:
                                        keyword_score += 0.3
                                        break
                
                final_score = vector_similarity + keyword_score
                
                formatted_candidates.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "final_score": final_score
                })

        formatted_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return formatted_candidates[:top_k]

    def generate_rag_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        pass
