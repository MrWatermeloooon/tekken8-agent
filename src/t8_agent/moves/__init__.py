"""Versioned full-roster move catalog and Tekken command notation."""

from .catalog import CATALOG_SCHEMA_VERSION, CompiledMoveCatalog, compile_catalog, load_compiled_catalog
from .notation import CommandSpec, InputStep, parse_command
from .legal import MoveRuntimeState, UNIVERSAL_ACTIONS, legal_action_mask

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CommandSpec",
    "CompiledMoveCatalog",
    "InputStep",
    "compile_catalog",
    "load_compiled_catalog",
    "parse_command",
]
from .catalog import CompiledMoveCatalog, compile_catalog, load_compiled_catalog
from .notation import CommandSpec, InputStep, parse_command
from .validation import CharacterGateReport, evaluate_character_gate

__all__ = [
    "CharacterGateReport", "CommandSpec", "CompiledMoveCatalog", "InputStep",
    "MoveRuntimeState", "UNIVERSAL_ACTIONS", "compile_catalog", "evaluate_character_gate",
    "legal_action_mask", "load_compiled_catalog", "parse_command",
]
