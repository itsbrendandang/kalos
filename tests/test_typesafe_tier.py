"""Tests for the TypeSafe tier of `kalos.normalize` (`typesafe_tier.py`), the
plan -> `ColumnRoles` seam (`roles.py`), and `/api/run`'s `roles=auto`.

Fully hermetic: no network call is made and no real TypeSafe key is used.
Most tests feed canned typed answers straight into `plan_from_answers` or
swap `_make_client` for a fake; the wire test drives the real `typesafe-sdk`
client against an in-process `httpx2.MockTransport` (skipped only when the
optional SDK is not installed).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalos.normalize import build_payload, load_config, plan_to_roles, propose_plan
from kalos.normalize import typesafe_tier
from kalos.normalize.plan import ColumnPlan, NormalizationPlan

# --- fixtures ------------------------------------------------------------------- #


def _messy_df() -> pd.DataFrame:
    """Same shape as the other normalize tests' fixture, plus a header that is
    not an exact synonym ("Glucose Feed") so a name question is asked."""
    return pd.DataFrame(
        {
            "Client Sample Name": ["Acme-1", "Acme-2", "Acme-3", "Acme-4", "Acme-5", "Acme-6"],
            "Operator": ["J. Rivera", "J. Rivera", "M. Chen", "M. Chen", "J. Rivera", "M. Chen"],
            "Campaign": ["C-100", "C-100", "C-100", "C-101", "C-101", "C-101"],
            "Titer (g/L)": [3.2, 3.6, 4.1, 3.9, 4.4, 4.0],
            "Temp": ["34.6 C", "34.8 C", "35.0 C", "34.9 C", "35.1 C", "34.7 C"],
            "Glucose Feed": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )


def _choice(choice: str, confidence: float, probabilities: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(choice=choice, confidence=confidence, probabilities=probabilities)


def _role(choice: str, confidence: float = 0.9) -> SimpleNamespace:
    rest = (1.0 - confidence) / 4
    probs = {r: rest for r in ("target", "feature", "group", "metadata", "freetext")}
    probs[choice] = confidence
    return _choice(choice, confidence, probs)


def _answers(payload: dict, roles: dict[str, SimpleNamespace], *, identity: dict[str, float] | None = None,
             names: dict[str, SimpleNamespace] | None = None) -> SimpleNamespace:
    """Canned answers keyed exactly like the real question names."""
    identity = identity or {}
    names = names or {}
    choices: dict[str, SimpleNamespace] = {}
    nouls: dict[str, SimpleNamespace] = {}
    for i, entry in enumerate(payload["columns"]):
        header = entry["header"]
        choices[f"role_{i}"] = roles[header]
        nouls[f"identity_{i}"] = SimpleNamespace(noul=identity.get(header, 0.02))
        if header in names:
            choices[f"name_{i}"] = names[header]
    return SimpleNamespace(choices=choices, nouls=nouls)


def _default_roles() -> dict[str, SimpleNamespace]:
    return {
        "Campaign": _role("group"),
        "Titer (g/L)": _role("target"),
        "Temp": _role("feature"),
        "Glucose Feed": _role("feature"),
    }


@pytest.fixture
def typesafe_env(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-not-real")
    monkeypatch.delenv("KALOS_TYPESAFE_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("KALOS_TYPESAFE_MODEL", raising=False)


# --- config ------------------------------------------------------------------------ #


def test_typesafe_provider_is_gated_on_its_key(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "typesafe")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-irrelevant")
    assert load_config().enabled_live is False
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-not-real")
    config = load_config()
    assert config.provider == "typesafe"
    assert config.enabled_live is True
    assert config.typesafe_model == "jev-latest"
    assert config.typesafe_min_confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.8", 0.8), ("7", 1.0), ("-1", 0.0), ("nan", 0.6), ("high", 0.6)]
)
def test_min_confidence_is_clamped_and_malformed_falls_back(monkeypatch, raw, expected):
    monkeypatch.setenv("KALOS_TYPESAFE_MIN_CONFIDENCE", raw)
    assert load_config().typesafe_min_confidence == pytest.approx(expected)


# --- questions ---------------------------------------------------------------------- #


def test_questions_reference_columns_by_path_and_ask_names_only_when_needed():
    payload, _ = build_payload(_messy_df())
    headers = [c["header"] for c in payload["columns"]]
    questions = typesafe_tier.build_questions(payload["columns"], range(len(headers)))

    for i, header in enumerate(headers):
        assert questions[f"role_{i}"]["type"] == "choice"
        assert set(questions[f"role_{i}"]["criteria"]) == {"target", "feature", "group", "metadata", "freetext"}
        assert questions[f"identity_{i}"]["type"] == "noul"
        # Question names never reach the model, so each instruction names its column.
        assert f"`columns[{i}]`" in questions[f"role_{i}"]["instructions"]
        assert repr(header) in questions[f"identity_{i}"]["instructions"]

    temp_i = headers.index("Temp")
    glucose_i = headers.index("Glucose Feed")
    # "Temp" is an exact alias, resolved by code; "Glucose Feed" is not.
    assert f"name_{temp_i}" not in questions
    name_q = questions[f"name_{glucose_i}"]
    assert "glucose_feed" in name_q["criteria"]  # the no-match outcome: keep own name
    assert "feed_rate" in name_q["criteria"]


def test_state_is_the_screened_payload_only():
    df = _messy_df()
    payload, dropped = build_payload(df)
    state = typesafe_tier.build_state(payload)
    dumped = json.dumps(state)
    assert set(dropped) == {"Client Sample Name", "Operator"}
    for raw_value in ("Acme-1", "J. Rivera", "M. Chen", "Client Sample Name", "Operator"):
        assert raw_value not in dumped


# --- composing the plan from typed answers ---------------------------------------------- #


def test_plan_from_confident_answers(typesafe_env):
    df = _messy_df()
    payload, dropped = build_payload(df)
    answers = _answers(
        payload,
        _default_roles(),
        names={"Glucose Feed": _choice("feed_rate", 0.85, {"feed_rate": 0.85, "glucose_feed": 0.15})},
    )
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, answers, load_config())

    assert plan.created_by == "typesafe"
    assert plan.model == "jev-latest"
    plan.validate()
    by_raw = {c.raw_name: c for c in plan.columns}
    assert [c.raw_name for c in plan.columns] == list(df.columns)  # sheet order kept
    assert by_raw["Client Sample Name"].role == "identity"
    assert by_raw["Operator"].role == "identity"
    assert by_raw["Campaign"].role == "group"
    assert by_raw["Titer (g/L)"].role == "target"
    # Units stay a deterministic rule: same token and suffix as the offline plan.
    assert by_raw["Temp"].unit_token == "C"
    assert by_raw["Temp"].canonical_name == "temperature_c"
    assert by_raw["Glucose Feed"].canonical_name == "feed_rate"
    assert "feature=0.90" in by_raw["Temp"].note  # the probabilities are auditable


def test_low_confidence_role_keeps_the_offline_guess(typesafe_env):
    df = _messy_df()
    payload, dropped = build_payload(df)
    roles = _default_roles()
    roles["Campaign"] = _role("feature", confidence=0.4)  # unsure, and wrong
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, _answers(payload, roles), load_config())
    campaign = {c.raw_name: c for c in plan.columns}["Campaign"]
    assert campaign.role == "group"  # the offline guess
    assert "kept offline guess" in campaign.note


def test_low_confidence_name_keeps_the_column_name(typesafe_env):
    df = _messy_df()
    payload, dropped = build_payload(df)
    answers = _answers(
        payload,
        _default_roles(),
        names={"Glucose Feed": _choice("feed_rate", 0.5, {"feed_rate": 0.5, "glucose_feed": 0.5})},
    )
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, answers, load_config())
    assert {c.raw_name: c for c in plan.columns}["Glucose Feed"].canonical_name == "glucose_feed"


def test_identity_judgment_drops_a_column_the_prescreen_missed(typesafe_env):
    df = _messy_df()
    payload, dropped = build_payload(df)
    answers = _answers(payload, _default_roles(), identity={"Campaign": 0.93})
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, answers, load_config())
    campaign = {c.raw_name: c for c in plan.columns}["Campaign"]
    assert campaign.role == "identity"
    assert campaign.canonical_name is None
    assert campaign.is_identity is True


def test_second_target_is_kept_as_metadata_never_a_feature(typesafe_env):
    df = _messy_df().assign(Purity=[0.91, 0.92, 0.9, 0.93, 0.95, 0.94])
    payload, dropped = build_payload(df)
    roles = _default_roles()
    roles["Titer (g/L)"] = _role("target", confidence=0.95)
    roles["Purity"] = _role("target", confidence=0.7)
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, _answers(payload, roles), load_config())
    by_raw = {c.raw_name: c for c in plan.columns}
    assert by_raw["Titer (g/L)"].role == "target"
    assert by_raw["Purity"].role == "metadata"
    assert "second outcome column" in by_raw["Purity"].note


def test_duplicate_canonical_names_are_resolved(typesafe_env):
    df = _messy_df().assign(**{"Feed Pump": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]})
    payload, dropped = build_payload(df)
    roles = {**_default_roles(), "Feed Pump": _role("feature")}
    both_feed = _choice("feed_rate", 0.9, {"feed_rate": 0.9})
    answers = _answers(payload, roles, names={"Glucose Feed": both_feed, "Feed Pump": both_feed})
    plan = typesafe_tier.plan_from_answers(df, payload, dropped, answers, load_config())
    plan.validate()
    by_raw = {c.raw_name: c for c in plan.columns}
    assert by_raw["Glucose Feed"].canonical_name == "feed_rate"
    assert by_raw["Feed Pump"].canonical_name == "feed_pump"


# --- the live path through propose_plan ------------------------------------------------- #


class _FakeClient:
    def __init__(self, respond):
        self.respond = respond
        self.calls: list[dict] = []

    def system_one(self, *, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return self.respond(state, questions)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_propose_plan_uses_typesafe_when_configured(typesafe_env, monkeypatch):
    df = _messy_df()
    payload, _ = build_payload(df)
    canned = _answers(payload, _default_roles(), names={"Glucose Feed": _choice("glucose_feed", 0.8, {})})
    fake = _FakeClient(lambda state, questions: canned)
    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: fake)

    plan = propose_plan(df)
    assert plan.created_by == "typesafe"
    assert len(fake.calls) == 1
    sent = json.dumps(fake.calls[0])
    for raw_value in ("Acme-1", "J. Rivera", "Client Sample Name"):
        assert raw_value not in sent


def test_wide_sheets_are_chunked_over_the_same_state(typesafe_env, monkeypatch):
    n = typesafe_tier._COLUMNS_PER_REQUEST + 5
    df = pd.DataFrame({f"x{k}": np.linspace(0, 1, 6) + k for k in range(n)}).assign(titer=np.arange(6.0))
    payload, _ = build_payload(df)
    roles = {f"x{k}": _role("feature") for k in range(n)} | {"titer": _role("target")}
    canned = _answers(payload, roles)
    fake = _FakeClient(lambda state, questions: canned)
    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: fake)

    plan = propose_plan(df)
    assert plan.created_by == "typesafe"
    assert len(fake.calls) == 2
    assert fake.calls[0]["state"] == fake.calls[1]["state"]
    asked = set(fake.calls[0]["questions"]) | set(fake.calls[1]["questions"])
    assert {f"role_{i}" for i in range(n + 1)} <= asked


def test_api_failure_falls_back_to_offline(typesafe_env, monkeypatch):
    def boom(state, questions):
        raise ConnectionError("simulated outage")

    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: _FakeClient(boom))
    plan = propose_plan(_messy_df())  # must not raise
    assert plan.created_by == "offline"


def test_missing_answer_falls_back_to_offline(typesafe_env, monkeypatch):
    fake = _FakeClient(lambda state, questions: SimpleNamespace(choices={}, nouls={}))
    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: fake)
    assert propose_plan(_messy_df()).created_by == "offline"


def test_without_a_key_typesafe_is_never_called(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "typesafe")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def never(config):
        raise AssertionError("TypeSafe client constructed without a key")

    monkeypatch.setattr(typesafe_tier, "_make_client", never)
    assert propose_plan(_messy_df()).created_by == "offline"


def test_real_sdk_wire_round_trip(typesafe_env, monkeypatch):
    """The real typesafe-sdk client accepts the question dictionaries this tier
    builds and its parsed response composes into a plan."""
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")

    df = _messy_df()
    roles = {"Campaign": "group", "Titer (g/L)": "target", "Temp": "feature", "Glucose Feed": "feature"}
    requests: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        headers = [c["header"] for c in body["state"]["columns"]]
        answers: dict[str, dict] = {}
        for name, question in body["questions"].items():
            kind, i = name.rsplit("_", 1)
            header = headers[int(i)]
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.03}
            elif kind == "role":
                pick = roles[header]
                probs = {label: (0.92 if label == pick else 0.02) for label in question["criteria"]}
                answers[name] = {"type": "choice", "choice": pick, "confidence": 0.9, "probabilities": probs}
            else:
                own = list(question["criteria"])[-1]
                answers[name] = {"type": "choice", "choice": own, "confidence": 0.8,
                                 "probabilities": {label: 0.0 for label in question["criteria"]} | {own: 1.0}}
        return httpx2.Response(200, json={"model": body["model"], "usage": {}, "answers": answers})

    def make_client(config):
        return sdk.TypeSafeClient(model=config.typesafe_model, transport=httpx2.MockTransport(handler))

    monkeypatch.setattr(typesafe_tier, "_make_client", make_client)
    plan = propose_plan(df)

    assert plan.created_by == "typesafe"
    assert requests and requests[0]["model"] == "jev-latest"
    by_raw = {c.raw_name: c for c in plan.columns}
    assert by_raw["Titer (g/L)"].role == "target"
    assert by_raw["Glucose Feed"].canonical_name == "glucose_feed"
    assert by_raw["Temp"].canonical_name == "temperature_c"


# --- plan -> ColumnRoles ------------------------------------------------------------- #


def test_plan_to_roles_maps_raw_names():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan("Operator", None, "identity", is_identity=True),
            ColumnPlan("Notes", None, "freetext", redacted=True),
            ColumnPlan("Campaign", "campaign", "group"),
            ColumnPlan("Lot", "lot", "group"),
            ColumnPlan("Run Date", "run_date", "metadata"),
            ColumnPlan("Titer (g/L)", "titer", "target"),
            ColumnPlan("Temp", "temperature_c", "feature", unit_token="C", to_base=True),
        ],
        created_by="typesafe",
        model="jev-latest",
    )
    roles = plan_to_roles(plan)
    assert roles is not None
    assert roles.target == "Titer (g/L)"
    assert roles.features == ("Temp",)
    assert roles.groups == "Campaign"
    assert set(roles.ids) == {"Operator", "Notes", "Run Date", "Lot"}


def test_plan_to_roles_without_a_target_is_none():
    plan = NormalizationPlan(columns=[ColumnPlan("Temp", "temperature", "feature")], created_by="offline")
    assert plan_to_roles(plan) is None


# --- /api/run roles=auto ---------------------------------------------------------------- #


def test_decide_roles_uses_the_typesafe_plan(typesafe_env, monkeypatch):
    pytest.importorskip("fastapi")
    from kalos.portal.app import _decide_roles

    df = _messy_df()
    payload, _ = build_payload(df)
    canned = _answers(payload, _default_roles(), names={"Glucose Feed": _choice("glucose_feed", 0.8, {})})
    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: _FakeClient(lambda s, q: canned))

    roles, decision = _decide_roles(df, None)
    assert roles is not None
    assert roles.target == "Titer (g/L)"
    assert roles.groups == "Campaign"
    assert decision["applied"] is True
    assert decision["created_by"] == "typesafe"
    # The decision lists column names and rationale, never a cell value.
    assert "Acme-1" not in json.dumps(decision)

    overridden, _ = _decide_roles(df, "Glucose Feed")
    assert overridden is not None
    assert overridden.target == "Glucose Feed"
    assert "Glucose Feed" not in overridden.features


def test_decide_roles_falls_back_when_no_consistent_plan(monkeypatch):
    pytest.importorskip("fastapi")
    from kalos.portal.app import _decide_roles

    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    # Two outcome-looking columns: the offline plan refuses two targets.
    df = pd.DataFrame({"titer": [1.0, 2.0, 3.0], "purity": [0.9, 0.8, 0.7], "ph": [6.5, 6.8, 7.0]})
    roles, decision = _decide_roles(df, None)
    assert roles is None
    assert decision["applied"] is False


def test_run_upload_with_auto_roles_end_to_end(typesafe_env, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("botorch")
    import kalos.portal.app as portal

    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")

    rng = np.random.default_rng(3)
    n = 30
    temp = rng.uniform(30, 37, n)
    feed = rng.uniform(1, 4, n)
    df = pd.DataFrame({
        "Campaign": rng.choice(["C-1", "C-2", "C-3"], n),
        "Temp": [f"{t:.1f} C" for t in temp],
        "Glucose Feed": feed.round(3),
        "Titer (g/L)": (2 + 0.4 * feed - 0.05 * (temp - 34) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })
    payload, _ = build_payload(df)
    canned = _answers(payload, _default_roles(), names={"Glucose Feed": _choice("glucose_feed", 0.8, {})})
    monkeypatch.setattr(typesafe_tier, "_make_client", lambda config: _FakeClient(lambda s, q: canned))

    result = portal._run_uploaded_sync(df.to_csv(index=False).encode(), None, False, "sheet.csv", "auto")
    assert result["target"] == "Titer (g/L)"
    assert result["role_decision"]["applied"] is True
    assert result["role_decision"]["created_by"] == "typesafe"
