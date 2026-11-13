# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Pure-VLA protocol, completed-cell, and native-outcome regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robots.robocasa.eval.summarize_pure_vla import (
    DEFAULT_MANIFEST,
    cell_spec,
    load_manifest,
    main,
    manifest_sha256,
    policy_seed,
    summarize_results,
    validate_cell_result,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _result(manifest, task="ScrubCuttingBoard", seed=1, *, success=False):
    spec = cell_spec(manifest, task, seed)
    steps = 8 if success else spec["official_horizon"]
    return {
        **spec,
        "schema_version": "1.0",
        "experiment_id": "pure_vla",
        "valid": True,
        "completed": True,
        "is_smoke": False,
        "effective_horizon": spec["official_horizon"],
        "native_steps": steps,
        "policy_calls": (steps + 7) // 8,
        "success": success,
        "any_success": success,
        "final_state_success": success,
        "first_success_native_step": 7 if success else None,
        "termination_reason": "success" if success else "horizon",
        "elapsed_s": 123.5,
        "source": {"rldx_commit": manifest["dependencies"]["rldx"]["commit"]},
        "initial_observation": {"task_language_sha256": "a" * 64},
        "seed_protocol": {"constructor_seed": seed},
        "input_audit": {"status": "ok"},
    }


def test_manifest_matches_baseline_task_seed_product_and_official_horizons():
    manifest = load_manifest()
    baseline = json.loads(
        (DEFAULT_MANIFEST.parents[1] / "robots/robocasa/eval/target50.json").read_text()
    )
    cells = []
    for group, split in manifest["splits"].items():
        assert split["tasks"] == baseline["splits"][group]["tasks"]
        assert split["seeds"] == [1, 2, 3, 4, 5]
        cells.extend((task, seed) for task in split["tasks"] for seed in split["seeds"])
    assert len(set(cells)) == manifest["total_cells"] == 160
    assert sum(manifest["task_horizons"][task] for task, _ in cells) == 262_000
    assert manifest["task_horizons"]["ScrubCuttingBoard"] == 800
    assert manifest["task_horizons"]["ArrangeBreadBasket"] == 2900
    assert manifest["scene_seed"]["python_hash_seed"] == 0


def test_policy_seeds_are_cell_specific_and_independent_of_dispatch_order():
    cells = [("ScrubCuttingBoard", 1), ("ArrangeTea", 1), ("ScrubCuttingBoard", 2)]
    first = {cell: policy_seed(*cell) for cell in cells}
    assert first == {cell: policy_seed(*cell) for cell in reversed(cells)}
    assert len(set(first.values())) == len(cells)
    assert all(0 <= value < 2**32 for value in first.values())
    with pytest.raises(ValueError, match="integer"):
        cell_spec(load_manifest(), "ScrubCuttingBoard", True)


def test_manifest_hash_ignores_formatting_but_detects_changed_protocol(tmp_path):
    manifest = load_manifest()
    path = _write(tmp_path / "compact.json", manifest)
    assert manifest_sha256(path) == manifest_sha256(DEFAULT_MANIFEST)
    changed = copy.deepcopy(manifest)
    changed["task_horizons"]["ScrubCuttingBoard"] += 8
    assert manifest_sha256(changed) != manifest_sha256(manifest)


@pytest.mark.parametrize(
    "corruption", ["duplicate_task", "duplicate_seed", "missing_horizon"]
)
def test_manifest_rejects_ambiguous_cells(tmp_path, corruption):
    manifest = load_manifest()
    split = manifest["splits"]["composite_seen"]
    if corruption == "duplicate_task":
        split["tasks"].append(split["tasks"][0])
    elif corruption == "duplicate_seed":
        split["seeds"].append(1)
    else:
        del manifest["task_horizons"]["ScrubCuttingBoard"]
    with pytest.raises(ValueError):
        load_manifest(_write(tmp_path / "manifest.json", manifest))


@pytest.mark.parametrize("success", [False, True])
def test_valid_native_outcome_can_resume_including_task_failure(tmp_path, success):
    manifest = load_manifest()
    path = _write(tmp_path / "result.json", _result(manifest, success=success))
    assert (
        validate_cell_result(
            path, manifest=manifest, expected_task="ScrubCuttingBoard", expected_seed=1
        )
        == []
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 2),
        ("seed", True),
        ("policy_seed", 0),
        ("protocol_sha256", "old-protocol"),
        ("completed", False),
        ("valid", False),
        ("is_smoke", True),
        ("native_steps", 801),
        ("native_steps", 799),
        ("effective_horizon", 320),
        ("policy_calls", 1),
        ("any_success", "false"),
        ("final_state_success", True),
        ("first_success_native_step", 7),
        ("termination_reason", "infrastructure_error"),
        ("termination_reason", []),
        ("input_audit", {"status": "failed"}),
        ("initial_observation", {}),
        ("source", None),
        ("elapsed_s", float("nan")),
    ],
)
def test_resume_rejects_mismatched_incomplete_or_inconsistent_result(
    tmp_path, key, value
):
    manifest = load_manifest()
    result = _result(manifest)
    result[key] = value
    path = _write(tmp_path / "result.json", result)
    assert validate_cell_result(
        path, manifest=manifest, expected_task="ScrubCuttingBoard", expected_seed=1
    )


def test_empty_directory_and_truncated_json_are_not_completed(tmp_path):
    manifest = load_manifest()
    path = tmp_path / "result.json"
    kwargs = {
        "manifest": manifest,
        "expected_task": "ScrubCuttingBoard",
        "expected_seed": 1,
    }
    assert validate_cell_result(path, **kwargs)
    path.write_text('{"success":true,')
    assert validate_cell_result(path, **kwargs)


def test_partial_final_chunk_is_accepted_without_rounding_horizon_up(tmp_path):
    manifest = load_manifest()
    result = _result(manifest, "DeliverStraw")
    assert result["native_steps"] == 1700
    assert result["policy_calls"] == 213
    path = _write(tmp_path / "result.json", result)
    assert not validate_cell_result(
        path, manifest=manifest, expected_task="DeliverStraw", expected_seed=1
    )


def test_transient_success_stays_separate_from_terminal_success(tmp_path):
    manifest = load_manifest()
    result = _result(manifest, success=True)
    result["final_state_success"] = False
    path = _write(tmp_path / "composite_seen/ScrubCuttingBoard_s1/result.json", result)
    assert not validate_cell_result(
        path, manifest=manifest, expected_task="ScrubCuttingBoard", expected_seed=1
    )
    summary, errors = summarize_results(tmp_path, manifest=manifest)
    assert not errors
    assert summary["complete"] is False
    assert summary["overall"]["valid"] == 1
    assert summary["overall"]["any_success"] == 1
    assert summary["overall"]["final_state_success"] == 0
    assert summary["overall"]["transient_success"] == 1


def test_summary_excludes_archived_infrastructure_attempts_and_flags_duplicates(
    tmp_path,
):
    manifest = load_manifest()
    result = _result(manifest, success=True)
    canonical = tmp_path / "composite_seen/ScrubCuttingBoard_s1/result.json"
    _write(canonical, result)
    _write(
        tmp_path / "quarantine/composite_seen/ScrubCuttingBoard_s1/result.json", result
    )
    _write(
        tmp_path / "attempts/composite_seen/ScrubCuttingBoard_s1/result.json", result
    )
    summary, errors = summarize_results(tmp_path, manifest=manifest)
    assert not errors
    assert summary["overall"]["valid"] == 1
    _write(tmp_path / "composite_seen/ScrubCuttingBoard_s1/old/result.json", result)
    summary, errors = summarize_results(tmp_path, manifest=manifest)
    assert summary["complete"] is False
    assert any("duplicate" in error for error in errors)


def test_full_manifest_counts_valid_failures_and_requires_all_160_cells(tmp_path):
    manifest = load_manifest()
    for group, split in manifest["splits"].items():
        for task in split["tasks"]:
            for seed in split["seeds"]:
                _write(
                    tmp_path / group / f"{task}_s{seed}" / "result.json",
                    _result(manifest, task, seed, success=seed == 1),
                )
    summary, errors = summarize_results(tmp_path, manifest=manifest)
    assert errors == []
    assert summary["complete"] is True
    assert summary["overall"]["valid"] == 160
    assert summary["overall"]["any_success"] == 32
    assert summary["overall"]["failure"] == 128
    assert summary["splits"]["composite_seen"]["any_success"] == 16
    assert main([str(tmp_path), "--output", str(tmp_path / "summary.json")]) == 0
    (tmp_path / "composite_unseen/ArrangeTea_s5/result.json").unlink()
    assert main([str(tmp_path)]) == 1


def _policy_observation():
    import numpy as np

    from robots.robocasa.eval.pure_vla import LANGUAGE_KEY, STATE_DIMS, VIDEO_KEYS

    return {
        **{key: np.zeros((1, 4, 256, 256, 3), dtype=np.uint8) for key in VIDEO_KEYS},
        **{
            key: np.zeros((1, 1, size), dtype=np.float32)
            for key, size in STATE_DIMS.items()
        },
        LANGUAGE_KEY: ["Wash the lettuce in the sink."],
    }


def test_terminal_score_is_cached_before_reset_and_preserves_transient_success(
    tmp_path,
):
    from robots.robocasa.eval.pure_vla import EpisodeRecord

    record = EpisodeRecord(tmp_path, horizon=8)
    for step in range(8):
        record.record_step(success=step == 3)
    terminal = record.freeze()
    assert terminal["any_success"] is True
    assert terminal["final_state_success"] is False
    assert terminal["first_success_native_step"] == 4
    assert terminal["native_steps"] == 8
    assert terminal["termination_reason"] == "success"
    # Cleanup/reset state must never replace the completed episode's score.
    record.any_success = False
    record.native_steps = 0
    assert record.freeze() == terminal
    with pytest.raises(RuntimeError, match="frozen"):
        record.record_step(success=False)


def test_early_exit_cannot_be_misclassified_as_valid_task_failure(tmp_path):
    from robots.robocasa.eval.pure_vla import EpisodeRecord

    record = EpisodeRecord(tmp_path, horizon=9)
    for _ in range(8):
        record.record_step(success=False)
    with pytest.raises(RuntimeError, match="without a valid terminal"):
        record.freeze()
    record.record_step(success=False)
    assert record.freeze()["termination_reason"] == "horizon"


def test_native_horizon_is_enforced_even_inside_an_action_chunk(tmp_path):
    from robots.robocasa.eval.pure_vla import EpisodeRecord

    record = EpisodeRecord(tmp_path, horizon=3)
    for _ in range(3):
        record.record_step(success=False)
    with pytest.raises(RuntimeError, match="budget"):
        record.record_step(success=False)


@pytest.mark.parametrize(
    "extra_key", ["success", "task_progress", "world_map", "grasp_contact", "memory"]
)
def test_vla_input_rejects_privileged_feedback_and_planner_memory(extra_key):
    from robots.robocasa.eval.pure_vla import audit_policy_observation

    observation = _policy_observation()
    assert audit_policy_observation(observation)["status"] == "ok"
    observation[extra_key] = True
    with pytest.raises(ValueError, match="Unexpected VLA observation"):
        audit_policy_observation(observation)


@pytest.mark.parametrize(
    "corruption", ["single_frame", "missing_state", "nan_state", "subtask"]
)
def test_vla_input_requires_original_modalities_history_and_complete_instruction(
    corruption,
):
    import numpy as np

    from robots.robocasa.eval.pure_vla import (
        LANGUAGE_KEY,
        STATE_DIMS,
        VIDEO_KEYS,
        audit_policy_observation,
    )

    observation = _policy_observation()
    original_language = observation[LANGUAGE_KEY][0]
    if corruption == "single_frame":
        observation[VIDEO_KEYS[0]] = observation[VIDEO_KEYS[0]][:, :1]
    elif corruption == "missing_state":
        del observation[next(iter(STATE_DIMS))]
    elif corruption == "nan_state":
        observation[next(iter(STATE_DIMS))][0, 0, 0] = np.nan
    else:
        observation[LANGUAGE_KEY] = ["Pick up the lettuce."]
    with pytest.raises(ValueError):
        audit_policy_observation(observation, original_language)


def test_policy_sampling_preserves_environment_rng_and_continuous_model_memory(
    tmp_path,
):
    import numpy as np

    from robots.robocasa.eval.pure_vla import AuditedPolicy, EpisodeRecord

    class FakePolicy:
        def __init__(self):
            self.options = []

        def get_action(self, observation, options):
            self.options.append(options)
            return {"sample": np.random.random()}, {}

    record = EpisodeRecord(tmp_path, horizon=16)
    policy = FakePolicy()
    audited = AuditedPolicy(policy, record, sampling_seed=123)
    observation = _policy_observation()
    outer_state = np.random.get_state()
    try:
        first, _ = audited.get_action(observation, {"reset_memory": [True]})
        second, _ = audited.get_action(observation, {"reset_memory": [False]})
        after_state = np.random.get_state()
        assert after_state[0] == outer_state[0]
        np.testing.assert_array_equal(after_state[1], outer_state[1])
        assert after_state[2:] == outer_state[2:]
        expected = np.random.RandomState(123).random_sample(2)
        assert [first["sample"], second["sample"]] == pytest.approx(expected)
        assert policy.options == [{"reset_memory": [True]}, {"reset_memory": [False]}]
        with pytest.raises(ValueError, match="policy memory"):
            audited.get_action(observation, {"reset_memory": [True]})
    finally:
        audited.close()


def test_policy_metadata_reads_the_nested_action_head_after_rollout():
    from types import SimpleNamespace

    import numpy as np

    from robots.robocasa.eval.pure_vla import policy_runtime_metadata

    # The real RLDX outer model has no num_inference_timesteps attribute.
    head = SimpleNamespace(
        num_inference_timesteps=4,
        action_horizon=16,
        denoising_timesteps=np.array([1.0, 0.75, 0.5, 0.25]),
    )
    core = SimpleNamespace(
        model=SimpleNamespace(action_model=head),
        use_memory=True,
        _rtc_enabled=False,
        rtc_inference_mode="off",
        rtc_inference_delay=0,
        rtc_exec_horizon=8,
    )
    metadata = policy_runtime_metadata(core)
    assert metadata["num_inference_timesteps"] == 4
    assert metadata["predicted_action_horizon"] == 16
    assert metadata["denoising_timesteps"] == [1.0, 0.75, 0.5, 0.25]
    assert metadata["use_memory"] is True
    assert metadata["rtc_enabled"] is False
    assert metadata["rtc_exec_horizon"] == 8


@pytest.mark.parametrize("shape", ["no_model", "no_action_head", "empty_action_head"])
def test_optional_policy_metadata_cannot_invalidate_a_completed_rollout(shape):
    from types import SimpleNamespace

    from robots.robocasa.eval.pure_vla import policy_runtime_metadata

    core = SimpleNamespace()
    if shape != "no_model":
        core.model = SimpleNamespace()
    if shape == "empty_action_head":
        core.model.action_model = SimpleNamespace()
    assert all(value is None for value in policy_runtime_metadata(core).values())
