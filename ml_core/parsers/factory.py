import os
from .base import BaseParser
from .code_splitter import PythonParser

class SplitterFactory:
    _parsers = {
        ".py": PythonParser()
    }

    @classmethod
    def get_parser(cls, file_path:str) -> BaseParser:
        _, ext = os.path.splitext(file_path)
        if ext not in cls._parsers:
            return ValueError(f"Расширение {ext} не поддерживается")
        return cls._parsers[ext]
    
    @classmethod
    def get_supported_extensions(cls):
        return list(cls._parsers.keys())
