"""Stateful SALUTE conversation helper for CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class SaluteConversation:
    """Maintain SALUTE prompt state and handle control commands."""

    location_prefill: str = ""
    fields: Sequence[Tuple[str, str]] = (
        ("size", "Suurus"),
        ("activity", "Tegevus"),
        ("location", "Asukoht"),
        ("unit", "Üksus"),
        ("time", "Aeg"),
        ("equipment", "Varustus"),
    )
    skip_words: Sequence[str] = ("vahele", "skip", "järgmine")
    rewind_word: str = "tagasi"
    answers: dict = field(init=False)
    index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.answers = {key: "" for key, _ in self.fields}
        if "location" in self.answers and self.location_prefill:
            self.answers["location"] = self.location_prefill

    @property
    def is_complete(self) -> bool:
        """Return whether every SALUTE field has been filled or skipped."""

        return self.index >= len(self.fields)

    @property
    def current_field(self) -> Tuple[str, str]:
        """Return the active field key/label pair."""

        if self.is_complete:
            raise IndexError("Conversation already complete.")
        return self.fields[self.index]

    def prompt_text(self) -> str:
        """Return the CLI prompt for the current field."""

        _, label = self.current_field
        return f"[SALUTE_PROMPT] Palun ütle {label}. (või 'vahele'/'tagasi')"

    def handle_response(self, response: str) -> None:
        """Process a user response and update the conversation state."""

        response = response or ""
        lowered = response.strip().lower()

        if lowered.startswith(self.rewind_word):
            self._rewind(lowered)
            return

        key, _ = self.current_field

        if lowered in self.skip_words:
            if key == "location" and self.answers.get(key):
                pass
            else:
                self.answers[key] = ""
            self.index += 1
            return

        self.answers[key] = response
        self.index += 1

    def _rewind(self, lowered_response: str) -> None:
        """Move the pointer to a previous field based on the response."""

        parts = lowered_response.split(maxsplit=1)
        if len(parts) == 1:
            self.index = max(0, self.index - 1)
            return

        target = parts[1]
        for pos, (key, label) in enumerate(self.fields):
            if target == key.lower() or target == label.lower():
                self.index = pos
                return

        self.index = max(0, self.index - 1)

    def report_lines(self) -> List[str]:
        """Return formatted SALUTE report lines in Estonian order."""

        return [f"{label}: {self.answers[key] or '–'}" for key, label in self.fields]


__all__ = ["SaluteConversation"]
