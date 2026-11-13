# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Validate completed pure-VLA cells and summarize both native success metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "experiments/pure_vla_composite.json"
)
GROUPS = ("composite_seen", "composite_unseen")


def manifest_sha256(manifest: Path | dict[str, Any]) -> str:
    """Hash canonical JSON so formatting does not change the protocol identity."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Read and validate the experiment's task, seed, and action-budget manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("protocol_id") != "robocasa-pure-vla-composite-v1":
        raise ValueError("unsupported pure-VLA protocol_id")
    if manifest.get("result_schema_version") != "1.0":
        raise ValueError("unsupported pure-VLA result schema")
    if manifest.get("environment_split") != "target":
        raise ValueError("pure-VLA composite requires the target environment split")
    if manifest.get("n_action_steps") != 8:
        raise ValueError("pure-VLA protocol requires 8 actions per prediction")
    if manifest.get("policy_seed", {}).get("namespace") != "rpent-pure-vla-v1":
        raise ValueError("unsupported policy seed namespace")
    if manifest["policy_seed"].get("algorithm") != "sha256-prefix-u32":
        raise ValueError("unsupported policy seed algorithm")
    if set(manifest.get("splits", {})) != set(GROUPS):
        raise ValueError("manifest must contain exactly Seen and Unseen composite")
    tasks: set[str] = set()
    cells = 0
    for group in GROUPS:
        split = manifest["splits"][group]
        names, seeds = split["tasks"], split["seeds"]
        if not names or len(names) != len(set(names)) or tasks.intersection(names):
            raise ValueError("task names must be unique across both splits")
        if any(not isinstance(t, str) or not t.isidentifier() for t in names):
            raise ValueError("task names must be identifiers")
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("seeds must be nonnegative integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seeds in manifest")
        if split["task_count"] != len(names):
            raise ValueError("split task_count does not match tasks")
        if split["cell_count"] != len(names) * len(seeds):
            raise ValueError("split cell_count does not match task/seed product")
        cells += split["cell_count"]
        tasks.update(names)
    if manifest["total_tasks"] != len(tasks) or manifest["total_cells"] != cells:
        raise ValueError("manifest totals do not match task/seed product")
    horizons = manifest["task_horizons"]
    if set(horizons) != tasks:
        raise ValueError("task horizons must cover exactly the selected tasks")
    if any(type(h) is not int or h < 1 for h in horizons.values()):
        raise ValueError("task horizons must be positive native-action counts")
    return manifest


def policy_seed(task_name: str, seed: int) -> int:
    """Choose a stable policy RNG seed independent of GPU or dispatch order."""
    message = f"rpent-pure-vla-v1:{task_name}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:4], "big")


def cell_spec(manifest: dict[str, Any], task_name: str, seed: int) -> dict[str, Any]:
    """Resolve one authorized task/seed cell without touching the simulator."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for group in GROUPS:
        split = manifest["splits"][group]
        if task_name in split["tasks"] and seed in split["seeds"]:
            return {
                "task_name": task_name,
                "seed": seed,
                "task_group": group,
                "split": manifest["environment_split"],
                "official_horizon": manifest["task_horizons"][task_name],
                "policy_seed": policy_seed(task_name, seed),
                "protocol_sha256": manifest_sha256(manifest),
            }
    raise ValueError(f"task/seed is not in the experiment manifest: {task_name}/{seed}")


def validate_cell_result(
    path: Path,
    *,
    manifest: dict[str, Any],
    expected_task: str,
    expected_seed: int,
    allow_smoke: bool = False,
) -> list[str]:
    """Return rejection reasons; an existing directory alone never counts as done."""
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = cell_spec(manifest, expected_task, expected_seed)
    except (OSError, ValueError, TypeError) as exc:
        return [f"cannot read completed result: {exc}"]
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    errors = []
    for key, value in {
        **expected,
        "schema_version": manifest["result_schema_version"],
        "experiment_id": manifest["experiment_id"],
        "valid": True,
        "completed": True,
    }.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            errors.append(f"{key} does not match the expected completed cell")
    smoke = result.get("is_smoke")
    if type(smoke) is not bool or (smoke and not allow_smoke):
        errors.append("smoke or unspecified run cannot count as a formal episode")
    horizon = result.get("effective_horizon")
    official = expected["official_horizon"]
    if type(horizon) is not int or horizon < 1 or horizon > official:
        errors.append("invalid effective_horizon")
    elif not smoke and horizon != official:
        errors.append("formal episode did not use the official task horizon")
    native_steps = result.get("native_steps")
    if (
        type(native_steps) is not int
        or native_steps < 1
        or type(horizon) is not int
        or native_steps > horizon
    ):
        errors.append("native_steps is outside the episode action budget")
    calls = result.get("policy_calls")
    if type(calls) is not int or calls < 1:
        errors.append("policy_calls must be a positive integer")
    elif type(native_steps) is int and not (calls <= native_steps <= calls * 8):
        errors.append("native_steps is inconsistent with 8-step policy calls")
    for name in ("success", "any_success", "final_state_success"):
        if type(result.get(name)) is not bool:
            errors.append(f"{name} must be a native boolean")
    any_success = result.get("any_success")
    if result.get("success") is not any_success:
        errors.append("success must equal the primary any_success metric")
    if result.get("final_state_success") is True and any_success is not True:
        errors.append("final success cannot be true without any_success")
    first = result.get("first_success_native_step")
    if any_success is True:
        if (
            type(first) is not int
            or type(native_steps) is not int
            or not 1 <= first <= native_steps
        ):
            errors.append("successful episode needs a valid first-success step")
    elif first is not None:
        errors.append("unsuccessful episode must not have a first-success step")
    reason = result.get("termination_reason")
    allowed = {"success", "horizon", "environment_terminated", "environment_truncated"}
    if not isinstance(reason, str) or reason not in allowed:
        errors.append("result has no valid rollout termination reason")
    if reason == "success" and any_success is not True:
        errors.append("success termination requires native any_success")
    if any_success is True and reason != "success":
        errors.append("native any_success must be recorded as success termination")
    if reason == "horizon" and native_steps != horizon:
        errors.append("horizon termination must exhaust the native-action budget")
    elapsed = result.get("elapsed_s")
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
        errors.append("elapsed_s must be finite and nonnegative")
    for name in ("source", "initial_observation", "seed_protocol"):
        if not isinstance(result.get(name), dict) or not result[name]:
            errors.append(f"missing {name} evidence")
    audit = result.get("input_audit")
    if not isinstance(audit, dict) or audit.get("status") != "ok":
        errors.append("policy input audit did not pass")
    return errors


def summarize_results(
    root: Path, *, manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Summarize canonical cells, excluding archived attempts and infrastructure errors."""
    root = Path(root)
    errors: list[str] = []
    summaries = {}
    accepted = []
    expected_paths: set[Path] = set()
    for group in GROUPS:
        split = manifest["splits"][group]
        group_rows = []
        missing = []
        rejected = []
        for task in split["tasks"]:
            for seed in split["seeds"]:
                path = root / group / f"{task}_s{seed}" / "result.json"
                expected_paths.add(path)
                label = f"{group}/{task}_s{seed}"
                if not path.is_file():
                    missing.append(label)
                    continue
                reasons = validate_cell_result(
                    path, manifest=manifest, expected_task=task, expected_seed=seed
                )
                if reasons:
                    rejected.append(label)
                    errors.extend(f"{label}: {reason}" for reason in reasons)
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                group_rows.append(row)
                accepted.append(row)
        summaries[group] = _count_rows(group_rows, split["cell_count"])
        summaries[group].update(missing=missing, rejected=rejected)
    for group in GROUPS:
        for path in sorted((root / group).rglob("result.json")):
            if path not in expected_paths:
                errors.append(
                    f"unexpected or duplicate result outside canonical cell: {path.relative_to(root)}"
                )
    summary = {
        "experiment_id": manifest["experiment_id"],
        "protocol_sha256": manifest_sha256(manifest),
        "splits": summaries,
        "overall": _count_rows(accepted, manifest["total_cells"]),
        "complete": len(accepted) == manifest["total_cells"] and not errors,
        "errors": errors,
    }
    return summary, errors


def _count_rows(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    successes = sum(row["any_success"] for row in rows)
    final = sum(row["final_state_success"] for row in rows)
    return {
        "expected": expected,
        "valid": len(rows),
        "any_success": successes,
        "final_state_success": final,
        "failure": len(rows) - successes,
        "any_success_rate": successes / len(rows) if rows else None,
        "final_state_success_rate": final / len(rows) if rows else None,
        "transient_success": sum(
            row["any_success"] and not row["final_state_success"] for row in rows
        ),
        "native_steps": sum(row["native_steps"] for row in rows),
        "policy_calls": sum(row["policy_calls"] for row in rows),
        "elapsed_s": round(sum(row["elapsed_s"] for row in rows), 3),
    }


def main(argv: list[str] | None = None) -> int:
    """Print a progress summary; exit zero only for a complete, validated suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary, _ = summarize_results(args.root, manifest=load_manifest(args.manifest))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot summarize pure-VLA results: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
