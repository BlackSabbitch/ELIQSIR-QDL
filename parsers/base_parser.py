# parsers/base_parser.py

from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, path):
        """Парсинг файла с диска. Возвращает (data, error_string)"""
        pass

    @abstractmethod
    def parse_stream(self, binary_content):
        """Парсинг бинарного контента из архива. Возвращает (data, error_string)"""
        pass

    @abstractmethod
    def _process_ligand(self, mol):
        """Обработка объекта молекулы лиганда"""
        pass

    @abstractmethod
    def _process_protein(self, path_or_bytes, is_file=True):
        """Обработка белка из пути или байтов"""
        pass
