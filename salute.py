"""Reusable SALUTE conversation engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from config import DEFAULT_SALUTE_FIELDS, SaluteField


@dataclass
class SaluteStepResult:
    """Outcome of handling a single SALUTE response."""

    completed: bool
    prompt: Optional[str] = None
    report_lines: Optional[Tuple[str, ...]] = None


@dataclass
class SaluteConversation:
    """Stateful helper that tracks SALUTE answers and progression."""

    fields: Sequence[SaluteField] = DEFAULT_SALUTE_FIELDS
    location: str = ""
    answers: Dict[str, str] = field(default_factory=dict)
    index: int = 0

    def __post_init__(self) -> None:
        for field in self.fields:
            self.answers.setdefault(field.key, "")
        if self.location:
            self.answers["location"] = self.location

    def set_location(self, location: str) -> None:
        """Update the location field with the current MGRS string."""

        self.answers["location"] = location

    def prompt(self) -> str:
        """Return the prompt for the current SALUTE field."""

        field = self.fields[self.index]
        return f"Palun ütle {field.label}. (või 'vahele'/'tagasi')"

    def handle(self, text: str) -> SaluteStepResult:
        """Process the user's response and advance or rewind as needed."""

        normalized = (text or "").strip()
        lowered = normalized.lower()

        if lowered.startswith("tagasi"):
            self._rewind(lowered)
            return SaluteStepResult(completed=False, prompt=self.prompt())

        field = self.fields[self.index]
        if lowered in {"vahele", "skip", "järgmine"}:
            self.answers[field.key] = ""
        else:
            self.answers[field.key] = normalized

        self.index += 1
        if self.index >= len(self.fields):
            return SaluteStepResult(completed=True, report_lines=self.render_report_lines())

        return SaluteStepResult(completed=False, prompt=self.prompt())

    def render_report_lines(self) -> Tuple[str, ...]:
        """Produce formatted SALUTE report lines."""

        return tuple(
            f"{field.label}: {self.answers.get(field.key) or '–'}" for field in self.fields
        )

    def _rewind(self, lowered: str) -> None:
        parts = lowered.split(maxsplit=1)
        if len(parts) == 2:
            target = parts[1]
            target_idx = self._find_field_index(target)
            if target_idx is not None:
                self.index = target_idx
                return
        self.index = max(self.index - 1, 0)

    def _find_field_index(self, token: str) -> Optional[int]:
        token = token.strip().lower()
        for idx, field in enumerate(self.fields):
            if token == field.key or token == field.label.lower():
                return idx
        return None


__all__ = ["SaluteConversation", "SaluteStepResult"]
