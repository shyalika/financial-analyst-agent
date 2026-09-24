import pytest

from src.utils.billing import TokenBudgetController


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.delenv("MAX_TOKEN_BUDGET_PER_RUN", raising=False)
    return TokenBudgetController()


# ---------------------------------------------------------------------------
# Cost calculation per pricing tier
# ---------------------------------------------------------------------------

def test_update_usage_calculates_cost_for_gpt_4o_mini(controller):
    # 1,000,000 prompt tokens @ $0.15/1M input = $0.15 exactly
    controller.max_budget_usd = 10.0
    cost = controller.update_usage_and_verify(model="gpt-4o-mini", prompt=1_000_000, completion=0)
    assert cost == pytest.approx(0.15)


def test_update_usage_calculates_cost_for_gpt_4o_with_completion_tokens(controller):
    controller.max_budget_usd = 100.0
    # 1M prompt @ $2.50 + 1M completion @ $10.00 = $12.50
    cost = controller.update_usage_and_verify(model="gpt-4o", prompt=1_000_000, completion=1_000_000)
    assert cost == pytest.approx(12.50)


def test_update_usage_falls_back_to_default_model_pricing_for_unknown_model(controller):
    controller.max_budget_usd = 10.0
    cost_known = controller.update_usage_and_verify(model="gpt-4o-mini", prompt=1_000_000, completion=0)
    controller.usage.total_cost_usd = 0.0
    cost_unknown = controller.update_usage_and_verify(model="some-unlisted-model", prompt=1_000_000, completion=0)
    assert cost_unknown == pytest.approx(cost_known)


# ---------------------------------------------------------------------------
# Cumulative tracking across calls
# ---------------------------------------------------------------------------

def test_usage_accumulates_across_multiple_calls(controller):
    controller.max_budget_usd = 10.0
    controller.update_usage_and_verify(model="gpt-4o-mini", prompt=500_000, completion=0)
    controller.update_usage_and_verify(model="gpt-4o-mini", prompt=500_000, completion=0)

    assert controller.usage.prompt_tokens == 1_000_000
    assert controller.usage.total_cost_usd == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Budget-cap enforcement
# ---------------------------------------------------------------------------

def test_crossing_budget_cap_raises_permission_error(controller):
    controller.max_budget_usd = 0.01
    with pytest.raises(PermissionError):
        controller.update_usage_and_verify(model="gpt-4o", prompt=1_000_000, completion=0)


def test_cost_under_cap_does_not_raise(controller):
    controller.max_budget_usd = 1.0
    controller.update_usage_and_verify(model="gpt-4o-mini", prompt=1000, completion=0)  # negligible cost


def test_budget_cap_is_checked_cumulatively_not_just_per_call(controller):
    controller.max_budget_usd = 0.20
    controller.update_usage_and_verify(model="gpt-4o-mini", prompt=1_000_000, completion=0)  # $0.15, under cap
    with pytest.raises(PermissionError):
        controller.update_usage_and_verify(model="gpt-4o-mini", prompt=1_000_000, completion=0)  # pushes to $0.30


# ---------------------------------------------------------------------------
# estimate_input_tokens
# ---------------------------------------------------------------------------

def test_estimate_input_tokens_returns_positive_int_for_nonempty_text(controller):
    count = controller.estimate_input_tokens("What was CBA's statutory NPAT for 1H26?")
    assert isinstance(count, int)
    assert count > 0


def test_estimate_input_tokens_longer_text_has_more_or_equal_tokens(controller):
    short = controller.estimate_input_tokens("NPAT")
    long = controller.estimate_input_tokens("NPAT " * 50)
    assert long >= short


def test_estimate_input_tokens_empty_string_is_zero(controller):
    assert controller.estimate_input_tokens("") == 0
