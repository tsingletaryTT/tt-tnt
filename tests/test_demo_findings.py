# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from demo_findings import FindingCard, load_findings


def test_load_findings_extracts_verdict_via_dotted_path(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"verdict": "PARTIAL"}))
    cards = [FindingCard("A", p, ("verdict",))]
    result = load_findings(cards)
    assert result == [{"title": "A", "verdict": "PARTIAL", "source": str(p)}]


def test_load_findings_walks_a_nested_dotted_path(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"outer": {"inner": "some verdict text"}}))
    cards = [FindingCard("B", p, ("outer", "inner"))]
    result = load_findings(cards)
    assert result[0]["verdict"] == "some verdict text"


def test_load_findings_skips_a_missing_file(tmp_path):
    cards = [FindingCard("Missing", tmp_path / "nope.json", ("verdict",))]
    assert load_findings(cards) == []


def test_load_findings_skips_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    cards = [FindingCard("Bad", p, ("verdict",))]
    assert load_findings(cards) == []


def test_load_findings_reports_none_when_path_does_not_resolve_to_a_string(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"verdict": {"nested": "dict, not a string"}}))
    cards = [FindingCard("C", p, ("verdict",))]
    result = load_findings(cards)
    assert result[0]["verdict"] is None


def test_default_finding_cards_is_a_nonempty_list():
    from demo_findings import FINDING_CARDS
    assert len(FINDING_CARDS) > 0
    assert all(isinstance(c, FindingCard) for c in FINDING_CARDS)
