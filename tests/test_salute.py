"""Tests for the SALUTE conversation control flow."""

from salute_conversation import SaluteConversation


def test_skip_advances_and_clears_answer():
    convo = SaluteConversation(location_prefill="58T DK 12345 67890")

    assert convo.current_field[0] == "size"

    convo.handle_response("vahele")

    assert convo.answers["size"] == ""
    assert convo.current_field[0] == "activity"


def test_location_prefill_is_preserved_on_skip():
    prefill = "58T DK 12345 67890"
    convo = SaluteConversation(location_prefill=prefill)

    assert convo.answers["location"] == prefill

    convo.handle_response("vahele")
    convo.handle_response("Patrull")

    assert convo.current_field[0] == "location"
    assert convo.answers["location"] == prefill

    convo.handle_response("vahele")

    assert convo.answers["location"] == prefill
    assert convo.current_field[0] == "unit"


def test_rewind_keyword_moves_back_one_field():
    convo = SaluteConversation(location_prefill="58T DK 12345 67890")

    convo.handle_response("Väike rühm")
    convo.handle_response("Valvamas")

    assert convo.current_field[0] == "location"

    convo.handle_response("tagasi")

    assert convo.current_field[0] == "activity"


def test_rewind_to_named_field():
    convo = SaluteConversation(location_prefill="58T DK 12345 67890")

    convo.handle_response("Rühm")
    convo.handle_response("Patrull")
    convo.handle_response("vahele")
    convo.handle_response("Kompanii")
    convo.handle_response("12:30")

    assert convo.current_field[0] == "equipment"

    convo.handle_response("tagasi aeg")

    assert convo.current_field[0] == "time"

    convo.handle_response("13:00")

    assert convo.answers["time"] == "13:00"
