import pytest

from embodied_agent_harness.language import instruction_task, target_for_instruction


def test_language_selects_the_task_and_visible_target():
    assert instruction_task("把字母汤罐放进篮子") == 0
    assert instruction_task("pick up the cream cheese and place it in the basket") == 1
    assert target_for_instruction("把牛奶放进篮子", {"milk_1": [], "basket_1": []}) == (
        "milk_1",
        "basket_1",
    )


@pytest.mark.parametrize(
    "text",
    [
        "把牛奶放在篮子旁边",
        "不要把牛奶放进篮子",
        "put milk on the basket",
        "put milk and butter in the basket",
        "open the drawer",
        "",
    ],
)
def test_unsupported_goal_is_not_silently_scored_as_another_goal(text):
    with pytest.raises(ValueError):
        instruction_task(text)


def test_instruction_must_match_the_benchmark_oracle(tmp_path):
    from embodied_agent_harness.benchmark import load_panel
    from embodied_agent_harness.spec import write_json

    panel = {
        "schema": "eah.benchmark.v1",
        "observation_mode": "state",
        "max_steps": 600,
        "max_skill_calls": 3,
        "cases": [
            {
                "id": "case-0",
                "split": "development",
                "task_id": 0,
                "layout_mode": "expanded_reset",
                "layout_seed": 42,
                "jitter_m": 0.05,
                "instruction": "把牛奶放进篮子",
            }
        ],
    }
    path = tmp_path / "panel.json"
    write_json(path, panel)
    with pytest.raises(ValueError, match="differs"):
        load_panel(path)
