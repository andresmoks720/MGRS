"""Tests covering SALUTE conversation skip and rewind flows."""

from __future__ import annotations

import pytest

from salute import SaluteConversation, SaluteStepResult


@pytest.fixture()
def conversation() -> SaluteConversation:
    """Return a fresh conversation using the default SALUTE fields."""

    return SaluteConversation()


def complete(conversation: SaluteConversation, *answers: str) -> SaluteStepResult:
    """Feed answers sequentially and return the last step result."""

    result: SaluteStepResult | None = None
    for answer in answers:
        result = conversation.handle(answer)
    assert result is not None
    return result


def test_vahele_skips_current_field(conversation: SaluteConversation) -> None:
    """"vahele" leaves the current field empty and advances."""

    result = conversation.handle("vahele")

    expected_prompt = (
        f"Palun ütle {conversation.fields[1].label}. (või 'vahele'/'tagasi')"
    )
    assert not result.completed
    assert result.prompt == expected_prompt

    final = complete(
        conversation,
        "Vaadeldav tegevus",
        "Sihtasukoht",
        "Üksus",
        "Ajahetk",
        "Varustus",
    )

    assert final.completed
    assert final.report_lines is not None
    assert final.report_lines[0] == "Suurus: –"


def test_tagasi_with_target_rewinds_to_field(conversation: SaluteConversation) -> None:
    """Targeted tagasi rewinds to the requested field prompt."""

    complete(conversation, "Esialgne suurus", "Esialgne tegevus")
    result = conversation.handle("tagasi tegevus")

    expected_prompt = (
        f"Palun ütle {conversation.fields[1].label}. (või 'vahele'/'tagasi')"
    )
    assert not result.completed
    assert result.prompt == expected_prompt

    final = complete(
        conversation,
        "Uuendatud tegevus",
        "Täpsustatud asukoht",
        "Üksus",
        "Ajahetk",
        "Varustus",
    )

    assert final.completed
    assert final.report_lines is not None
    assert final.report_lines[1] == "Tegevus: Uuendatud tegevus"
    assert final.report_lines[2] == "Asukoht: Täpsustatud asukoht"


def test_tagasi_without_target_rewinds_previous(conversation: SaluteConversation) -> None:
    """Plain tagasi rewinds to the immediate previous field."""

    conversation.handle("Algne suurus")
    result = conversation.handle("tagasi")

    expected_prompt = (
        f"Palun ütle {conversation.fields[0].label}. (või 'vahele'/'tagasi')"
    )
    assert not result.completed
    assert result.prompt == expected_prompt

    final = complete(
        conversation,
        "Uuendatud suurus",
        "Tegevus",
        "Asukoht",
        "Üksus",
        "Aeg",
        "Varustus",
    )

    assert final.completed
    assert final.report_lines is not None
    assert final.report_lines[0] == "Suurus: Uuendatud suurus"


def test_constructor_location_prefills_report() -> None:
    """Providing location to the constructor should pre-populate the report line."""

    conversation = SaluteConversation(location="34VEM12345678")

    final = complete(
        conversation,
        "Suurus",
        "Tegevus",
        "Asukoht käsitsi",
        "Üksus",
        "Aeg",
        "Varustus",
    )

    assert final.completed
    assert final.report_lines is not None
    assert final.report_lines[2] == "Asukoht: Asukoht käsitsi"
