# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Run one composite episode through the pinned official RLDX rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger("pure_vla")

VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_DIMS = {
    "state.end_effector_position_relative": 3,
    "state.end_effector_rotation_relative": 4,
    "state.gripper_qpos": 2,
    "state.base_position": 3,
    "state.base_rotation": 4,
}
LANGUAGE_KEY = "annotation.human.task_description"
BACKBONE_REVISION = "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one result or progress document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _array_sha256(array: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _save_rgb_montage(path: Path, observation: dict, *, batched: bool) -> None:
    """Save already observed RGB without an extra render or simulator step."""
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        observation[key][0, -1] if batched else observation[key] for key in VIDEO_KEYS
    ]
    imageio.imwrite(path, np.concatenate(frames, axis=1), quality=85)


def audit_policy_observation(
    observation: dict[str, Any], expected_language: str | None = None
) -> dict[str, Any]:
    """Validate the complete, unchanged checkpoint input at the policy boundary."""
    import numpy as np

    expected = set(VIDEO_KEYS) | set(STATE_DIMS) | {LANGUAGE_KEY}
    if set(observation) != expected:
        raise ValueError(
            f"Unexpected VLA observation keys: {set(observation) ^ expected}"
        )
    language = list(observation[LANGUAGE_KEY])
    if len(language) != 1 or not isinstance(language[0], str) or not language[0]:
        raise ValueError("Expected one nonempty native task instruction")
    if expected_language is not None and language[0] != expected_language:
        raise ValueError("Native task instruction changed during the episode")
    fields = {}
    for key in VIDEO_KEYS:
        value = np.asarray(observation[key])
        if value.shape != (1, 4, 256, 256, 3) or value.dtype != np.uint8:
            raise ValueError(f"Invalid video input {key}: {value.shape}/{value.dtype}")
        fields[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "frame_sha256": [_array_sha256(frame) for frame in value[0]],
        }
    for key, dimension in STATE_DIMS.items():
        value = np.asarray(observation[key])
        if value.shape != (1, 1, dimension) or not np.isfinite(value).all():
            raise ValueError(f"Invalid proprioception input {key}: {value.shape}")
        fields[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": value.tolist(),
        }
    return {
        "status": "ok",
        "task_language": language[0],
        "keys": sorted(expected),
        "fields": fields,
    }


class EpisodeRecord:
    """Count native actions and preserve the terminal result before vector reset."""

    def __init__(self, output_dir: Path, horizon: int):
        self.output_dir = Path(output_dir)
        self.horizon = horizon
        self.native_steps = 0
        self.any_success = False
        self.final_state_success = False
        self.first_success_native_step = None
        self.native_terminated = False
        self.native_truncated = False
        self.policy_calls = 0
        self.reset_calls = 0
        self.native_reset_calls = 0
        self.terminal: dict[str, Any] | None = None
        self.initial: dict[str, Any] = {}
        self.input_audit: dict[str, Any] = {}
        self.control_frequency_hz: float | None = None
        self.started = time.time()

    def record_step(
        self, *, success: bool, terminated: bool = False, truncated: bool = False
    ) -> None:
        """Record a native simulator step; frozen episodes cannot accept actions."""
        if self.terminal is not None:
            raise RuntimeError("Cannot execute actions after the episode was frozen")
        self.native_steps += 1
        if self.native_steps > self.horizon:
            raise RuntimeError("Official rollout exceeded the native action budget")
        self.final_state_success = bool(success)
        self.any_success |= bool(success)
        if success and self.first_success_native_step is None:
            self.first_success_native_step = self.native_steps
        self.native_terminated |= bool(terminated)
        self.native_truncated |= bool(truncated)

    def freeze(self) -> dict[str, Any]:
        """Cache the score once, before any reset or environment destruction."""
        if self.terminal is None:
            if self.any_success:
                reason = "success"
            elif self.native_steps >= self.horizon:
                reason = "horizon"
            elif self.native_terminated:
                reason = "environment_terminated"
            elif self.native_truncated:
                reason = "environment_truncated"
            else:
                raise RuntimeError("Episode ended without a valid terminal condition")
            self.terminal = {
                "any_success": self.any_success,
                "success": self.any_success,
                "final_state_success": self.final_state_success,
                "first_success_native_step": self.first_success_native_step,
                "native_steps": self.native_steps,
                "policy_calls": self.policy_calls,
                "termination_reason": reason,
            }
        return dict(self.terminal)

    def progress(self, phase: str) -> None:
        atomic_json(
            self.output_dir / "progress.json",
            {
                "phase": phase,
                "updated_at": time.time(),
                "native_steps": self.native_steps,
                "horizon": self.horizon,
                "policy_calls": self.policy_calls,
                "any_success": self.any_success,
                "elapsed_s": time.time() - self.started,
            },
        )


@contextmanager
def _environment_rng(seed: int):
    """Keep environment construction independent of model sampling state."""
    import numpy as np

    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _save_initial(native_env: Any, observation: dict, record: EpisodeRecord) -> None:
    import imageio.v2 as imageio
    import numpy as np

    directory = record.output_dir / "initial"
    directory.mkdir(exist_ok=True)
    images = {}
    for key in VIDEO_KEYS:
        name = key.removeprefix("video.") + ".png"
        imageio.imwrite(directory / name, observation[key])
        images[key] = {
            "artifact": f"initial/{name}",
            "array_sha256": _array_sha256(observation[key]),
            "file_sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
        }
    simulator_state = np.asarray(native_env.sim.get_state().flatten())
    np.save(directory / "simulator_state.npy", simulator_state)
    model_xml = native_env.sim.model.get_xml()
    (directory / "model.xml").write_text(model_xml, encoding="utf-8")
    state = {key: _jsonable(observation[key]) for key in STATE_DIMS}
    record.initial = {
        "task_language": observation[LANGUAGE_KEY],
        "robot_state": state,
        "rgb": images,
        "simulator_state_sha256": _array_sha256(simulator_state),
        "model_xml_sha256": hashlib.sha256(model_xml.encode()).hexdigest(),
        "episode_metadata": _jsonable(native_env.get_ep_meta()),
    }
    atomic_json(directory / "observation.json", record.initial)


def _make_recorded_env(record: EpisodeRecord, task_name: str, seed: int):
    """Adapt the current native Gym API without a second initial scene reset."""
    import gymnasium as gym
    import numpy as np
    from robocasa.wrappers import gym_wrapper

    original_create = gym_wrapper.create_env
    constructor_raw = {}
    native_instance = []
    original_native_reset = []

    def create_and_capture(**kwargs):
        env = original_create(**kwargs)
        native_instance.append(env)
        original_reset = env.reset
        original_native_reset.append(original_reset)

        def capture_reset(*args, **reset_kwargs):
            raw = original_reset(*args, **reset_kwargs)
            record.native_reset_calls += 1
            constructor_raw.update(
                {key: np.array(value, copy=True) for key, value in raw.items()}
            )
            return raw

        env.reset = capture_reset
        return env

    gym_wrapper.create_env = create_and_capture
    try:
        with _environment_rng(seed):
            base = gym_wrapper.RoboCasaGymEnv(
                env_name=task_name,
                split="target",
                seed=seed,
                camera_widths=256,
                camera_heights=256,
                camera_depths=False,
            )
    except BaseException:
        for env in native_instance:
            env.close()
        raise
    finally:
        gym_wrapper.create_env = original_create
        for env, reset in zip(native_instance, original_native_reset, strict=True):
            env.reset = reset
    if record.native_reset_calls != 1 or not constructor_raw:
        base.close()
        raise RuntimeError("Native Gym constructor reset protocol changed")
    first_observation = base.get_observation(constructor_raw)
    record.control_frequency_hz = float(base.env.control_freq)
    try:
        _save_initial(base.env, first_observation, record)
    except BaseException:
        base.close()
        raise
    trajectory = (record.output_dir / "native_steps.jsonl").open("w", encoding="utf-8")

    class RecordedEnv(gym.Wrapper):
        def __init__(self):
            super().__init__(base)
            self.last_observation = first_observation
            self.closed = False

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                raise ValueError("Reset seed overrides are forbidden in this protocol")
            record.reset_calls += 1
            if record.native_steps and record.terminal is None:
                record.freeze()
                _save_rgb_montage(
                    record.output_dir / "terminal.jpg",
                    self.last_observation,
                    batched=False,
                )
                atomic_json(
                    record.output_dir / "terminal.json",
                    {
                        **record.terminal,
                        "robot_state": {
                            key: _jsonable(self.last_observation[key])
                            for key in STATE_DIMS
                        },
                        "simulator_state_sha256": _array_sha256(
                            self.env.env.sim.get_state().flatten()
                        ),
                    },
                )
            # This object serves exactly one episode. Vector/rollout cleanup can
            # reset it, but must not sample another scene or overwrite its score.
            return self.last_observation, {"success": False}

        def step(self, action):
            action_copy = {key: _jsonable(value) for key, value in action.items()}
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.last_observation = observation
            record.record_step(
                success=bool(info["success"]),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            trajectory.write(
                json.dumps(
                    {
                        "native_step": record.native_steps,
                        "action": action_copy,
                        "success": bool(info["success"]),
                        "robot_state": {
                            key: _jsonable(observation[key]) for key in STATE_DIMS
                        },
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if record.native_steps % 8 == 0:
                trajectory.flush()
                record.progress("running")
            return observation, reward, terminated, truncated, info

        def close(self):
            if self.closed:
                return
            self.closed = True
            if not trajectory.closed:
                trajectory.flush()
                os.fsync(trajectory.fileno())
                trajectory.close()
            self.env.close()

    return RecordedEnv()


class AuditedPolicy:
    """Record inputs/outputs while leaving official inference and memory intact."""

    def __init__(self, policy: Any, record: EpisodeRecord, sampling_seed: int):
        import numpy as np

        self.policy = policy
        self.record = record
        self.python_state = random.Random(sampling_seed).getstate()
        self.numpy_state = np.random.RandomState(sampling_seed).get_state()
        self.language = None
        self.stream = (record.output_dir / "policy_calls.jsonl").open(
            "w", encoding="utf-8"
        )
        self.reset_count = 0

    def reset(self, *args, **kwargs):
        self.reset_count += 1
        return self.policy.reset(*args, **kwargs)

    def get_action(self, observation, options=None):
        import numpy as np

        audit = audit_policy_observation(observation, self.language)
        self.language = audit["task_language"]
        self.record.policy_calls += 1
        expected_reset = [self.record.policy_calls == 1]
        if options is None or options.get("reset_memory") != expected_reset:
            raise ValueError(
                "Official rollout did not preserve per-episode policy memory"
            )
        if not self.record.input_audit:
            self.record.input_audit = audit
        image_name = f"frames/policy_{self.record.policy_calls:04d}.jpg"
        _save_rgb_montage(
            self.record.output_dir / image_name, observation, batched=True
        )
        python_state, numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        started = time.monotonic()
        self.record.progress("inference")
        try:
            actions, info = self.policy.get_action(observation, options=options)
        finally:
            self.python_state, self.numpy_state = (
                random.getstate(),
                np.random.get_state(),
            )
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.stream.write(
            json.dumps(
                {
                    "policy_call": self.record.policy_calls,
                    "native_steps_before": self.record.native_steps,
                    "reset_memory": options["reset_memory"],
                    "rgb_artifact": image_name,
                    "input": audit,
                    "actions": _jsonable(actions),
                    "inference_s": time.monotonic() - started,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self.stream.flush()
        return actions, info

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def _dependency_versions() -> dict[str, Any]:
    versions = {}
    for name in (
        "rlinf-rldx",
        "rlinf-robocasa",
        "rpent-robocasa365",
        "robosuite",
        "gymnasium",
        "torch",
    ):
        try:
            package = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        source = json.loads(package.read_text("direct_url.json") or "{}")
        versions[name] = {
            "version": package.version,
            "commit": source.get("vcs_info", {}).get("commit_id"),
        }
    return versions


def policy_runtime_metadata(core_policy: Any) -> dict[str, Any]:
    """Read inference settings from RLDX's action head, not the outer model."""
    model = getattr(core_policy, "model", None)
    action_model = getattr(model, "action_model", None)
    timesteps = getattr(action_model, "num_inference_timesteps", None)
    horizon = getattr(action_model, "action_horizon", None)
    return {
        "num_inference_timesteps": None if timesteps is None else int(timesteps),
        "predicted_action_horizon": None if horizon is None else int(horizon),
        "denoising_timesteps": _jsonable(
            getattr(action_model, "denoising_timesteps", None)
        ),
        "use_memory": getattr(core_policy, "use_memory", None),
        "rtc_enabled": getattr(core_policy, "_rtc_enabled", None),
        "rtc_inference_mode": getattr(core_policy, "rtc_inference_mode", None),
        "rtc_inference_delay": getattr(core_policy, "rtc_inference_delay", None),
        "rtc_exec_horizon": getattr(core_policy, "rtc_exec_horizon", None),
    }


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    """Run a single cell, keeping the upstream rollout loop and policy intact."""
    from robots.robocasa.eval.summarize_pure_vla import (
        cell_spec,
        load_manifest,
        manifest_sha256,
        policy_seed,
    )

    manifest = load_manifest(args.manifest)
    spec = cell_spec(manifest, args.task_name, args.seed)
    official_horizon = manifest["task_horizons"][args.task_name]
    horizon = (
        official_horizon
        if args.smoke_max_native_steps is None
        else args.smoke_max_native_steps
    )
    if not 1 <= horizon <= official_horizon:
        raise ValueError(
            "Smoke horizon must be between 1 and the official task horizon"
        )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(
            "Output already contains result.json; use a fresh attempt directory"
        )
    sample_seed = policy_seed(args.task_name, args.seed)
    record = EpisodeRecord(output, horizon)
    result = {
        "schema_version": "1.0",
        "experiment_id": "pure_vla",
        "task_name": args.task_name,
        "seed": args.seed,
        "split": "target",
        "task_group": spec["task_group"],
        "official_horizon": official_horizon,
        "effective_horizon": horizon,
        "is_smoke": args.smoke_max_native_steps is not None,
        "protocol_sha256": manifest_sha256(manifest),
        "policy_seed": sample_seed,
        "valid": False,
        "completed": False,
    }
    audited = None
    environments = []
    started = time.time()
    record.progress("loading_model")
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")
        if args.cuda_device != 0:
            raise ValueError(
                "Use CUDA_VISIBLE_DEVICES to select a GPU; --cuda-device must be local 0"
            )
        if os.environ.get("RLDX_RESET_SEED"):
            raise ValueError(
                "Unset RLDX_RESET_SEED: this experiment forbids reset seed overrides"
            )
        import numpy as np
        import torch
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.eval import rollout_policy as official

        from robots.robocasa.vla_server import _normalize_legacy_processor_geometry

        policy = official.create_rldx_sim_policy(
            str(Path(args.checkpoint).resolve()),
            EmbodimentTag.GENERAL_EMBODIMENT,
            "",
            None,
            backbone_revision=args.backbone_revision,
        )
        geometry_normalized = _normalize_legacy_processor_geometry(
            policy.policy.processor
        )
        modality = policy.get_modality_config()
        video_indices = np.asarray(modality["video"].delta_indices)
        state_indices = np.asarray(modality["state"].delta_indices)
        if video_indices.tolist() != [-6, -4, -2, 0] or state_indices.tolist() != [0]:
            raise ValueError(
                "Checkpoint history configuration differs from the pinned protocol"
            )
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        audited = AuditedPolicy(policy, record, sample_seed)
        configs = official.WrapperConfigs(
            video=official.VideoConfig(video_dir=None),
            multistep=official.MultiStepConfig(
                video_delta_indices=video_indices,
                state_delta_indices=state_indices,
                n_action_steps=8,
                max_episode_steps=horizon,
                terminate_on_success=True,
            ),
        )

        def create_eval_env(
            *,
            env_idx,
            env_name,
            total_n_envs,
            wrapper_configs,
            start_episode_id=0,
            seed=42,
            robocasa_split="target",
        ):
            if (
                env_idx != 0
                or total_n_envs != 1
                or seed != args.seed
                or robocasa_split != "target"
            ):
                raise ValueError(
                    "Unexpected official rollout environment configuration"
                )
            if env_name != f"robocasa/{args.task_name}" or start_episode_id != 0:
                raise ValueError("Each invocation must run one fresh task/seed episode")
            record.progress("creating_environment")
            env = _make_recorded_env(record, args.task_name, args.seed)
            environments.append(env)
            return official.MultiStepWrapper(
                env,
                video_delta_indices=wrapper_configs.multistep.video_delta_indices,
                state_delta_indices=wrapper_configs.multistep.state_delta_indices,
                n_action_steps=wrapper_configs.multistep.n_action_steps,
                max_episode_steps=wrapper_configs.multistep.max_episode_steps,
                terminate_on_success=wrapper_configs.multistep.terminate_on_success,
            )

        original_factory = official.create_eval_env
        official.create_eval_env = create_eval_env
        try:
            rollout_result = official.run_rollout_gymnasium_policy(
                env_name=f"robocasa/{args.task_name}",
                policy=audited,
                wrapper_configs=configs,
                n_episodes=1,
                n_envs=1,
                video_dir=None,
                seed=args.seed,
                robocasa_split="target",
            )
        finally:
            official.create_eval_env = original_factory
        terminal = record.freeze()
        if (
            len(rollout_result[1]) != 1
            or bool(rollout_result[1][0]) != terminal["any_success"]
        ):
            raise RuntimeError(
                "Official rollout score disagrees with native-step audit"
            )
        if audited.reset_count != 1:
            raise RuntimeError("Unexpected number of policy resets")
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        result.update(terminal)
        core_policy = policy.policy
        result.update(
            {
                "valid": True,
                "completed": True,
                "initial_observation": record.initial,
                "input_audit": record.input_audit,
                "seed_protocol": {
                    "constructor_seed": args.seed,
                    "reset_seed": None,
                    "native_reset_calls": record.native_reset_calls,
                    "wrapper_reset_calls": record.reset_calls,
                    "first_reset_reuses_constructor_observation": True,
                    "global_environment_rng_seed": args.seed,
                    "baseline_global_rng_equivalence": "not_claimed",
                    "video_reset_seed_override": False,
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                },
                "source": {
                    "git_commit": source_commit,
                    "dependencies": _dependency_versions(),
                },
                "policy": {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_revision": manifest["dependencies"]["checkpoint"][
                        "revision"
                    ],
                    "backbone_revision": args.backbone_revision,
                    "video_delta_indices": video_indices.tolist(),
                    "state_delta_indices": state_indices.tolist(),
                    "n_action_steps": 8,
                    "reset_count": audited.reset_count,
                    "legacy_processor_geometry_normalized": geometry_normalized,
                    **policy_runtime_metadata(core_policy),
                },
                "control_frequency_hz": record.control_frequency_hz,
                "simulation_time_s": record.native_steps / record.control_frequency_hz,
            }
        )
        record.progress("completed")
    except BaseException as exc:
        result.update(
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "native_steps": record.native_steps,
                "policy_calls": record.policy_calls,
                "any_success": record.any_success,
                "final_state_success": record.final_state_success,
                "termination_reason": "infrastructure_error",
            }
        )
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        record.progress("infrastructure_error")
        raise
    finally:
        try:
            for environment in environments:
                environment.close()
            if audited is not None:
                audited.close()
        except BaseException as exc:
            result.update(
                {
                    "valid": False,
                    "completed": False,
                    "termination_reason": "infrastructure_error",
                    "cleanup_error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            raise
        finally:
            result["elapsed_s"] = time.time() - started
            result["completed_at"] = time.time()
            atomic_json(output / "result.json", result)
    return result


def main() -> int:
    """CLI for one cell; GPU scheduling is deliberately outside this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", "--am-checkpoint", required=True)
    parser.add_argument("--backbone-revision", default=BACKBONE_REVISION)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "experiments/pure_vla_composite.json",
    )
    parser.add_argument("--smoke-max-native-steps", type=int)
    args = parser.parse_args()
    try:
        result = run_episode(args)
    except Exception:
        logger.exception("Pure VLA episode failed")
        return 1
    logger.info(
        "Pure VLA result: %s",
        json.dumps(
            {
                key: result[key]
                for key in (
                    "task_name",
                    "seed",
                    "success",
                    "final_state_success",
                    "native_steps",
                    "elapsed_s",
                )
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
