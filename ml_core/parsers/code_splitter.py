import ast
import tree_sitter_languages
from typing import List
from .base import BaseParser
from tree_sitter import Parser, Node
from ..schemas.code_chunk import CodeChunk

class PythonParser(BaseParser):
    def parse_file(self, file_path) -> List[CodeChunk]:
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return chunks
        
        lines = code.splitlines()

        method_to_class = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_to_class[sub_node] = node.name

        for walk in ast.walk(tree):
            if isinstance(walk, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                element_type = "class" if isinstance(walk, ast.ClassDef) else "function"
                
                start = walk.lineno
                end = getattr(walk, "end_lineno", start) 
                
                chunk_content = "\n".join(lines[start - 1:end])
                docstring = ast.get_docstring(walk)

                display_name = walk.name
                if element_type == "function" and walk in method_to_class:
                    display_name = f"{method_to_class[walk]}.{walk.name}"

                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    element_type=element_type,
                    name=display_name,
                    start_line=start,
                    end_line=end,
                    docstring=docstring
                ))
                
        return chunks
    

class OtherlanguageParser(BaseParser):
    """Универсальный парсер для других языков"""
    def __init__(self, language: str):
        self.lang_name = language
        self.language = tree_sitter_languages.get_language(language)
        self.parser = Parser()
        self.parser.set_language(self.language)

        if language == "javascript":
            self.target_types = {
                "function_declaration": "function",
                "method_definition": "function",
                "class_declaration": "class",
                "arrow_function": "function"
            }
        elif language == "java":
            self.target_types = {
                "method_declaration": "function",
                "class_declaration": "class",
                "interface_declaration": "class"
            }
        else:
            raise ValueError(f"Язык {language} пока не поддерживается через Tree-Sitter")

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_bytes = f.read().encode("utf-8")
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
            return chunks

        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node

        self._traverse_tree(root_node, code_bytes, file_path, chunks)
        return chunks

    def _traverse_tree(self, node: Node, code_bytes: bytes, file_path: str, chunks: List[CodeChunk]):
        """Рекурсивный обход всех узлов дерева синтаксиса"""
        
        if node.type in self.target_types:
            element_type = self.target_types[node.type]
            
            name = "anonymous"
            name_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
            if name_node:
                name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")

            content = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            docstring = self._extract_docstring(node, code_bytes)

            chunks.append(CodeChunk(
                content=content,
                file_path=file_path,
                element_type=element_type,
                name=name,
                start_line=start_line,
                end_line=end_line,
                docstring=docstring
            ))

        for child in node.children:
            self._traverse_tree(child, code_bytes, file_path, chunks)

    def _extract_docstring(self, node: Node, code_bytes: bytes) -> str:
        """Вспомогательный метод для поиска комментариев над функцией/классом"""
        docstring = ""
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type in ["comment", "block_comment", "expression_statement"]:
            text = code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode("utf-8")
            if text.startswith("/**") or text.startswith("//"):
                docstring = text
        return docstring
