# parsers/_base_parser.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, path: str) -> Tuple[Optional[Any], Optional[str]]:
        """Parse a file from disk and return parsed data and an optional error."""
        pass

    @abstractmethod
    def _process_ligand(self, mol: Any) -> Tuple[Optional[Any], Optional[str]]:
        """Process a ligand object into parser-specific output."""
        pass

    @abstractmethod
    def _process_protein(self, path_or_bytes: Any, is_file: bool = True) -> Tuple[Optional[Any], Optional[str]]:
        """Process protein data from a path or byte stream."""
        pass
