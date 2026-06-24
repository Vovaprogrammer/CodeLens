from typing import Optional
from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    content: str = Field(..., description="Сам код")
    file_path: str = Field(..., description="Путь к файлу")
    element_type: str = Field(..., description="Тип: функция, метод, класс")
    name: str = Field(..., description="Имя инстанса")
    start_line: int = Field(..., description="Начальная строка в исходном файле")
    end_line: int = Field(..., description="Конечная строка в исходном файле")
    docstring: Optional[str] = Field(None, description="Документация к коду")
