from abc import ABC, abstractmethod
from typing import List
from ..schemas.code_chunk import CodeChunk

class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, file_path: str) -> List[CodeChunk]:
        """Принимает путь к файлу, возвращает список распарсенных чанков"""
        pass
