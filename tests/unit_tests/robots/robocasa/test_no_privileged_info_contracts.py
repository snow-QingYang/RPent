# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Combined observation restrictions through VLA, tool dispatch, and saved state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.robocasa import tools
from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.robot_spec import get_robot_spec
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.session import EnvState
from rpent.tools.common import CommonTools

_POSE_FIELDS = {
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "eef",
    "base_pos",
    "start_pos",
    "final_dist",
    "base_moved",
    "moved",
    "peak_lift",
    "base_drift",
}
_INTERNAL_HIDDEN_FIELDS = _POSE_FIELDS | {
    "task_progress",
    "grasped",
    "grasp_detected",
    "grasp_contact",
    "grasp_obj",
}
_PLANNER_HIDDEN_FIELDS = _INTERNAL_HIDDEN_FIELDS | {
    "robot0_gripper_qpos",
    "gripper_qpos",
    "held_apart",
    "ok",
    "stuck",
    "status",
    "stage",
    "steps",
    "executed_steps",
    "steps_applied",
    "chunks",
    "vla_desync",
    "elapsed_s",
    "agent_elapsed_s",
    "prompt",
    "base_clip",
    "video",
    "reset",
    "effective_prompt",
    "effective_max_chunks",
    "effective_n_action_steps",
    "effective_settle_patience",
    "prompt_overridden",
    "requested_prompt",
}
_TASK_LANGUAGE = "Scrub the cutting board with the sponge."


class _Env:
    """CPU environment boundary with observable steps and forbidden contact RPC."""

    def __init__(self) -> None:
        self.task_name = "ScrubCuttingBoard"
        self.actions: list[np.ndarray] = []
        self.reset_calls = 0
        self.progress_queries = 0
        self.success_after: int | None = None
        self.progress = {
            "board_contact_timer": 6,
            "sweep_range": 0.2,
            "gripper_obj_far": False,
        }
        self.eef_pos = np.array([0.2, 0.0, 0.9])
        self.eef_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper_qpos = np.array([0.02, 0.02])
        self.current_raw_obs = {
            "language": _TASK_LANGUAGE,
            "robot0_gripper_qpos": self.gripper_qpos,
            "robot0_base_pos": np.zeros(3),
            "robot0_base_quat": self.eef_quat,
            "robot0_base_to_eef_pos": self.eef_pos,
            "robot0_base_to_eef_quat": self.eef_quat,
        }

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return self.current_raw_obs

    @property
    def action_specs(self) -> dict[str, Any]:
        return {"default": {"low": [-1.0] * 12, "high": [1.0] * 12}}

    def validate_action(self, values: list[float]) -> np.ndarray:
        assert len(values) == 12
        return np.asarray(values, dtype=np.float64)

    def step(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())
        self.eef_pos[0] += 0.01

    def check_success(self) -> bool:
        return (
            self.success_after is not None and len(self.actions) >= self.success_after
        )

    @property
    def terminated(self) -> bool:
        return self.check_success()

    def grasp_contact(self) -> tuple[bool, str | None]:
        raise AssertionError("VLA execution queried simulator contact ground truth")

    def get_task_language(self) -> str:
        return _TASK_LANGUAGE

    def get_task_progress(self) -> dict[str, Any]:
        self.progress_queries += 1
        return self.progress

    def get_success_criteria_text(self) -> str:
        raise AssertionError("The toolkit requested simulator success-criterion source")

    def render_camera(
        self,
        camera_name: str,
        height: int | None = None,
        width: int | None = None,
        depth: bool = False,
    ) -> Any:
        # Encode simulation time so a repeated VLA call can expose history resets.
        rgb = np.full((4, 4, 3), len(self.actions), dtype=np.uint8)
        return (rgb, np.ones((4, 4))) if depth else rgb

    def world_map(self, camera_name: str) -> np.ndarray:
        return np.zeros((4, 4, 3))

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        return {"camera": camera_name, "height": 4, "width": 4}

    def reassemble_env_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                action["action.end_effector_position"],
                action["action.end_effector_rotation"],
                action["action.gripper_close"],
                action["action.base_motion"],
                action["action.control_mode"],
            ]
        )


class _Vla:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.reset_calls = 0

    def get_modality_config(self) -> dict[str, Any]:
        return {"video_delta_indices": [-2, 0], "hist_maxlen": 4}

    def predict(
        self, observation: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        self.calls.append((observation, options))
        return {
            "action.end_effector_position": np.zeros((1, 2, 3)),
            "action.end_effector_rotation": np.zeros((1, 2, 3)),
            "action.gripper_close": np.ones((1, 2, 1)),
            "action.base_motion": np.full((1, 2, 4), 0.8),
            "action.control_mode": np.ones((1, 2, 1)),
        }

    def reset_session(self) -> None:
        self.reset_calls += 1


class _ControllerEnv(_Env):
    """Advance a simple arm/base plant from actual native controller commands."""

    def __init__(self) -> None:
        super().__init__()
        self.current_raw_obs["robot0_base_pos"] = np.array([1.2, -0.7, 0.1])
        self.base_blocked = False
        self.arm_blocked = False
        self.yaw = 0.0

    def step(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())
        if not self.arm_blocked:
            self.eef_pos = self.eef_pos + 0.05 * action[:3]
        if not self.base_blocked:
            self.yaw += 0.1 * action[9]
            forward = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
            lateral = np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])
            self.current_raw_obs["robot0_base_pos"] = self.current_raw_obs[
                "robot0_base_pos"
            ] + 0.05 * (action[7] * forward + action[8] * lateral)
            self.current_raw_obs["robot0_base_quat"] = np.array(
                [0.0, 0.0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)]
            )
        self.current_raw_obs["robot0_base_to_eef_pos"] = (
            self.eef_pos - self.current_raw_obs["robot0_base_pos"]
        )


@pytest.fixture
def execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        "RLDX_MAX_CHUNKS",
        "RLDX_ACTION_STEPS_PER_CHUNK",
        "RLDX_SETTLE_PATIENCE",
        "RLDX_VIDEO_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    env, vla = _Env(), _Vla()
    primitives = RoboCasaPrimitives(
        env, tmp_path / "work", hi_res=0, vla_client=vla, allow_reset=False
    )
    # Bypass only the simulator-specific conversion import; run() and its
    # observation construction, control loop, stopping and history remain real.
    primitives._rldx._unmap = lambda action: action
    return primitives, env, vla


@pytest.fixture
def controller(tmp_path: Path):
    env = _ControllerEnv()
    primitives = RoboCasaPrimitives(
        env, tmp_path / "controller", hi_res=0, vla_client=_Vla(), allow_reset=False
    )
    primitives._rldx._unmap = lambda action: action
    return primitives, env


@pytest.fixture
def dispatcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_corpus, execution):
    # Reuse execution's deterministic VLA environment settings.
    del execution
    env, vla = _ControllerEnv(), _Vla()
    root = make_corpus(tmp_path / "dispatcher-corpus")
    output = tmp_path / "dispatcher"
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": str(output)}
    )
    memory = RoboCasaMemoryManager(
        TaskMemory.load(root, "OpenDrawer"), output_dir=output
    )
    robot = RoboCasaToolkit(
        runtime_kwargs={
            "env_client": env,
            "vla_client": vla,
            "workdir": tmp_path / "dispatcher-work",
            "hi_res": 0,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        state_output_dir=output,
        enable_direct_action=True,
    )
    robot._primitives._rldx._unmap = lambda action: action
    return robot, env, vla, output


def _call(primitives: RoboCasaPrimitives, method: str, **kwargs: Any) -> dict[str, Any]:
    return getattr(primitives, method)(
        prompt=_TASK_LANGUAGE,
        max_chunks=kwargs.pop("max_chunks", 1),
        n_action_steps=kwargs.pop("n_action_steps", 1),
        settle_patience=kwargs.pop("settle_patience", 999),
        **kwargs,
    ).data


def _assert_no_internal_hidden_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert not (_INTERNAL_HIDDEN_FIELDS & value.keys())
        for child in value.values():
            _assert_no_internal_hidden_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_internal_hidden_fields(child)


def _assert_no_planner_feedback(value: Any) -> None:
    if isinstance(value, dict):
        assert not (_PLANNER_HIDDEN_FIELDS & value.keys())
        for key, child in value.items():
            if key != "command":
                _assert_no_planner_feedback(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_planner_feedback(child)


@pytest.mark.parametrize("method,base_motion", [("rldx_skill", 0.8), ("rldx_arm", 0.1)])
def test_native_vla_keeps_gripper_feedback_without_querying_contact(
    execution, method: str, base_motion: float
) -> None:
    primitives, env, vla = execution

    result = _call(primitives, method)

    _assert_no_internal_hidden_fields(result)
    assert result["held_apart"] is True
    assert result["gripper_qpos"] == 0.02
    assert result["status"] == "cap"
    assert result["steps_applied"] == 1
    assert len(vla.calls) == len(env.actions) == 1
    np.testing.assert_allclose(env.actions[0][7:11], base_motion)


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
def test_planner_pose_restriction_preserves_full_vla_proprioception(
    execution, method: str
) -> None:
    primitives, env, vla = execution
    raw = env.current_raw_obs
    raw["robot0_base_pos"] = np.array([1.3, -0.4, 0.1])
    raw["robot0_base_quat"] = np.array([0.0, 0.0, 0.6, 0.8])
    raw["robot0_base_to_eef_pos"] = np.array([0.14, 0.26, 0.38])
    raw["robot0_base_to_eef_quat"] = np.array([0.6, 0.0, 0.0, 0.8])
    expected = {
        "state.gripper_qpos": raw["robot0_gripper_qpos"].copy(),
        "state.base_position": raw["robot0_base_pos"].copy(),
        "state.base_rotation": raw["robot0_base_quat"].copy(),
        "state.end_effector_position_relative": raw["robot0_base_to_eef_pos"].copy(),
        "state.end_effector_rotation_relative": raw["robot0_base_to_eef_quat"].copy(),
    }

    result = _call(primitives, method)

    _assert_no_internal_hidden_fields(result)
    observation, options = vla.calls[0]
    assert options == {"reset_memory": [True]}
    assert {key for key in observation if key.startswith("state.")} == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(observation[key], value[None, None, :])
    assert observation["annotation.human.task_description"] == [_TASK_LANGUAGE]


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
def test_native_success_still_ends_the_vla_loop(execution, method: str) -> None:
    primitives, env, vla = execution
    env.success_after = 1

    result = _call(primitives, method, max_chunks=5)

    _assert_no_internal_hidden_fields(result)
    assert result["status"] == "success"
    assert result["chunks"] == result["steps_applied"] == 1
    assert len(vla.calls) == len(env.actions) == 1


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
@pytest.mark.parametrize(
    "settle_eps,status,chunks", [(0.012, "settled", 2), (0.005, "cap", 5)]
)
def test_hidden_eef_motion_still_controls_vla_settling(
    execution, method: str, settle_eps: float, status: str, chunks: int
) -> None:
    primitives, env, vla = execution

    result = _call(
        primitives,
        method,
        max_chunks=5,
        settle_patience=2,
        settle_eps=settle_eps,
    )

    _assert_no_internal_hidden_fields(result)
    assert result["status"] == status
    assert result["chunks"] == result["steps_applied"] == chunks
    assert len(vla.calls) == len(env.actions) == chunks


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
def test_consecutive_calls_preserve_policy_memory_and_frame_history(
    execution, method: str
) -> None:
    primitives, env, vla = execution

    _call(primitives, method)
    _call(primitives, method)

    assert [options["reset_memory"] for _, options in vla.calls] == [[True], [False]]
    assert env.reset_calls == 1  # Construction only.
    assert vla.reset_calls == 0
    second_video = vla.calls[1][0]["video.robot0_agentview_left"]
    assert second_video[0, :, 0, 0, 0].tolist() == [0, 1]
    assert primitives._vla_desync is False


@pytest.mark.parametrize("success", [False, True])
def test_current_state_exposes_no_proprioception(execution, success):
    primitives, env, _ = execution
    env.success_after = 0 if success else None

    state = primitives.current_state_dict()

    _assert_no_planner_feedback(state)
    assert env.progress_queries == 0
    assert state["success"] is success
    assert state["robocasa_terminated"] is success
    assert state["state"] == {}


def test_saved_states_and_history_do_not_reintroduce_hidden_feedback(
    execution, tmp_path: Path
) -> None:
    primitives, _, _ = execution
    run_dir = tmp_path / "run"
    state = EnvState(run_dir)
    for method in ("rldx_skill", "rldx_arm"):
        result = _call(primitives, method)
        tools.dump_state(
            primitives,
            state,
            log={"command": {"name": method}, "result": result, "elapsed_s": 0.1},
        )

    # Exercise the actual latest and historical state tools, including image IO.
    for requested_step, expected_step in ((None, 1), (0, 0), (1, 1)):
        response = tools.view_env_state(step=requested_step, state=state)
        _assert_no_planner_feedback(response.data)
        assert response.data["step"] == expected_step
        assert response.data["log"]["result"] == {}
        assert response.data["state"] == {}
        assert len(response.data["images"]) == 3

    persisted = json.loads((run_dir / "states.json").read_text())
    for record in persisted["steps"]:
        _assert_no_planner_feedback(record)
    assert len(persisted["steps"]) == 2
    for record in persisted["steps"]:
        assert record["result"] == {}
        assert record["state"] == {}


@pytest.mark.parametrize("requested_step", [None, 0, 1])
def test_historical_feedback_is_filtered_without_changing_agent_commands(
    tmp_path: Path, requested_step: int | None
) -> None:
    state = EnvState(tmp_path / "legacy")
    feedback = dict.fromkeys(_PLANNER_HIDDEN_FIELDS, 0.5)
    command = {
        "action": "move_to",
        "xyz": [1.1, -0.4, 0.9],
        "steps": 17,
        "max_chunks": 5,
        "agent_notes": {"base_pos": [1.0, -0.4, 0.0], "final_dist": 0.3},
    }
    result = {
        **feedback,
        "ok": False,
        "stuck": True,
        "status": "settled",
        "held_apart": True,
        "gripper_qpos": 0.02,
        "children": [{"result": {**feedback, "ok": True}}],
    }
    for success in (False, True):
        with state.record_step(
            state={
                **feedback,
                "robot0_gripper_qpos": [0.02, 0.02],
                "nested": [{**feedback, "gripper_qpos": 0.02}],
            },
            command=command,
            result=result,
            terminated=success,
            elapsed_s=1.2,
            extras={
                "success": success,
                "task_language": _TASK_LANGUAGE,
                "vla_desync": True,
            },
        ):
            pass

    response = tools.view_env_state(step=requested_step, state=state)

    assert not response.is_error
    _assert_no_planner_feedback(response.data)
    assert response.data["state"] == {"nested": [{}]}
    assert response.data["log"]["command"] == command
    assert response.data["log"]["result"] == {
        "children": [{"result": {}}],
    }
    assert response.data["success"] is (requested_step != 0)
    assert response.data["robocasa_terminated"] is (requested_step != 0)
    assert state.get(0).command == command
    assert state.get(0).result == result


def test_dump_state_filters_nested_legacy_results_before_persistence(
    execution, tmp_path
):
    primitives, _, _ = execution
    state = EnvState(tmp_path / "persisted")
    command = {"action": "move_to", "xyz": [0.1, 0.2, 0.3], "steps": 7}
    legacy = {
        "ok": True,
        "result": [dict.fromkeys(_PLANNER_HIDDEN_FIELDS, 0.5)],
    }

    tools.dump_state(primitives, state, log={"command": command, "result": legacy})

    persisted = json.loads((tmp_path / "persisted/states.json").read_text())
    for record in persisted["steps"]:
        _assert_no_planner_feedback(record)
    assert persisted["steps"][0]["result"] == {
        "result": [{}],
    }
    assert persisted["steps"][0]["command"] == command
    assert _PLANNER_HIDDEN_FIELDS <= legacy["result"][0].keys()


@pytest.mark.parametrize("method", ["move_to", "move_delta"])
@pytest.mark.parametrize("max_steps,expected_ok", [(80, True), (1, False)])
def test_arm_controller_uses_hidden_pose_and_returns_only_allowed_feedback(
    controller, method: str, max_steps: int, expected_ok: bool
) -> None:
    primitives, env = controller
    displacement = np.array([0.12, 0.02, 0.03])
    target = env.eef_pos.copy() + displacement

    result = getattr(primitives, method)(
        (target if method == "move_to" else displacement).tolist(),
        max_steps=max_steps,
    ).data

    _assert_no_internal_hidden_fields(result)
    assert result["ok"] is expected_ok
    assert result["gripper_qpos"] == 0.02
    assert result["steps"] == len(env.actions) - 9  # Three calibration probes.
    assert result["steps"] > 0
    assert bool(np.linalg.norm(env.eef_pos - target) < 0.012) is expected_ok
    assert primitives._vla_desync is True


@pytest.mark.parametrize(
    "blocked,max_steps,expected_ok,expected_stuck",
    [(False, 40, True, None), (False, 4, False, False), (True, 4, False, True)],
)
def test_navigation_uses_hidden_base_pose_for_control_and_stuck_detection(
    controller,
    blocked: bool,
    max_steps: int,
    expected_ok: bool,
    expected_stuck: bool | None,
) -> None:
    primitives, env = controller
    env.base_blocked = blocked
    start = env.current_raw_obs["robot0_base_pos"].copy()
    target = start[:2] + [0.8, 0.0]

    result = primitives.navigate_to(target.tolist(), tol=0.08, max_steps=max_steps).data

    _assert_no_internal_hidden_fields(result)
    assert result["ok"] is expected_ok
    assert result.get("stuck") is expected_stuck
    assert result["steps"] == len(env.actions) - 6  # Forward calibration.
    assert result["steps"] > 0
    distance = np.linalg.norm(env.current_raw_obs["robot0_base_pos"][:2] - target)
    assert bool(distance < 0.08) is expected_ok
    if blocked:
        np.testing.assert_array_equal(env.current_raw_obs["robot0_base_pos"], start)
    else:
        assert env.current_raw_obs["robot0_base_pos"][0] > start[0]
    assert primitives._pos_jac is None


@pytest.mark.parametrize(
    "method,kwargs,steps",
    [
        ("rotate_pitch", {"target_pitch": 0.6, "n": 3}, 3),
        ("set_gripper", {"gripper": 1, "steps": 3}, 3),
        ("release", {"steps": 3}, 3),
        ("move_base", {"forward": 0.4, "steps": 3}, 3),
    ],
)
def test_manual_actions_execute_without_returning_pose(
    controller, method: str, kwargs: dict[str, Any], steps: int
) -> None:
    primitives, env = controller

    result = getattr(primitives, method)(**kwargs).data

    _assert_no_internal_hidden_fields(result)
    assert result["ok"] is True
    assert len(env.actions) == steps
    assert primitives._vla_desync is True
    if method == "rotate_pitch":
        assert all(action[3] > 0 for action in env.actions)
    elif method == "move_base":
        assert all(action[7] == 0.4 for action in env.actions)
    else:
        assert result["gripper_qpos"] == [0.02, 0.02]
        assert all(
            action[6] == (1 if method == "set_gripper" else -1)
            for action in env.actions
        )


@pytest.mark.parametrize("failed_stage", [None, "approach", "descent", "lift"])
def test_scripted_grasp_success_and_failure_results_hide_pose(
    controller, monkeypatch: pytest.MonkeyPatch, failed_stage: str | None
) -> None:
    primitives, env = controller
    move_to = primitives.move_to
    stages = iter(("approach", "descent", "lift"))

    def move_with_stage_failure(*args, **kwargs):
        if next(stages) == failed_stage:
            kwargs["max_steps"] = 0
        return move_to(*args, **kwargs)

    monkeypatch.setattr(primitives, "move_to", move_with_stage_failure)

    result = primitives.scripted_grasp([0.22, 0.0, 0.9]).data

    _assert_no_internal_hidden_fields(result)
    assert result["ok"] is (failed_stage is None)
    assert result.get("stage") == failed_stage
    assert result["gripper_qpos"] == ([0.02, 0.02] if failed_stage is None else 0.02)
    assert env.actions


@pytest.mark.parametrize("allow_reset", [False, True])
def test_reset_policy_and_calibration_reset_remain_without_pose(execution, allow_reset):
    primitives, env, vla = execution
    primitives._allow_reset = allow_reset
    primitives._pos_jac = np.eye(3)
    primitives._fwd_offset = 0.5

    result = primitives.reset().data

    _assert_no_internal_hidden_fields(result)
    if allow_reset:
        assert result == {"ok": True, "reset": True}
        assert env.reset_calls == 2
        assert vla.reset_calls == 1
        assert primitives._pos_jac is primitives._fwd_offset is None
    else:
        assert "DISABLED" in result["error"]
        assert env.reset_calls == 1
        assert vla.reset_calls == 0
        np.testing.assert_array_equal(primitives._pos_jac, np.eye(3))
        assert primitives._fwd_offset == 0.5


@pytest.mark.parametrize(
    "method,arguments,blocked,native_feedback",
    [
        ("move_to", {"xyz": [0.32, 0.02, 0.93], "max_steps": 80}, None, {"ok": True}),
        ("move_to", {"xyz": [0.32, 0.02, 0.93], "max_steps": 1}, None, {"ok": False}),
        ("move_delta", {"dxyz": [0.12, 0.02, 0.03]}, None, {"ok": True}),
        (
            "move_delta",
            {"dxyz": [0.12, 0.02, 0.03], "max_steps": 1},
            None,
            {"ok": False},
        ),
        ("rotate_pitch", {"n": 3}, None, {"ok": True}),
        ("set_gripper", {"steps": 3}, None, {"ok": True}),
        ("release", {"steps": 3}, None, {"ok": True}),
        ("move_base", {"forward": 0.4, "steps": 3}, None, {"ok": True}),
        (
            "navigate_to",
            {"xy": [2.0, -0.7], "tol": 0.08, "max_steps": 40},
            None,
            {"ok": True},
        ),
        (
            "navigate_to",
            {"xy": [2.0, -0.7], "tol": 0.08, "max_steps": 4},
            None,
            {"ok": False, "stuck": False},
        ),
        (
            "navigate_to",
            {"xy": [2.0, -0.7], "tol": 0.08, "max_steps": 4},
            "base",
            {"ok": False, "stuck": True},
        ),
        ("scripted_grasp", {"xyz": [0.22, 0.0, 0.9]}, None, {"ok": True}),
        (
            "scripted_grasp",
            {"xyz": [0.22, 0.0, 0.9]},
            "arm",
            {"ok": False, "stage": "approach"},
        ),
        (
            "rldx_skill",
            {"prompt": _TASK_LANGUAGE, "max_chunks": 1, "n_action_steps": 1},
            None,
            {"status": "cap"},
        ),
        (
            "rldx_arm",
            {"prompt": _TASK_LANGUAGE, "max_chunks": 1, "n_action_steps": 1},
            None,
            {"status": "cap"},
        ),
        (
            "rldx_skill",
            {"prompt": _TASK_LANGUAGE, "max_chunks": 5, "n_action_steps": 1},
            "success",
            {"status": "success"},
        ),
        (
            "rldx_arm",
            {"prompt": _TASK_LANGUAGE, "max_chunks": 5, "n_action_steps": 1},
            "success",
            {"status": "success"},
        ),
        (
            "rldx_skill",
            {"prompt": "Pick up the sponge.", "max_chunks": 1, "n_action_steps": 1},
            None,
            {"status": "cap", "prompt_overridden": True},
        ),
        ("execute_action", {"values": [0.1] * 12}, None, {"executed_steps": 1}),
        ("reset", {}, None, {"ok": True, "reset": True}),
    ],
)
def test_dispatch_hides_feedback_without_changing_native_control(
    dispatcher,
    controller,
    method: str,
    arguments: dict[str, Any],
    blocked: str | None,
    native_feedback: dict[str, Any],
) -> None:
    robot, env, _, output = dispatcher
    reference, reference_env = controller
    for plant in (env, reference_env):
        plant.base_blocked = blocked == "base"
        plant.arm_blocked = blocked == "arm"
        plant.success_after = 1 if blocked == "success" else None
    if method == "reset":
        robot._primitives._allow_reset = reference._allow_reset = True

    native = getattr(reference, method)(**arguments)
    response = robot.execute_tool(method, arguments)

    for key, expected in native_feedback.items():
        assert native.data[key] == expected
    assert not response.is_error
    _assert_no_planner_feedback(response.data)
    assert response.data["state"] == {}
    assert response.data["log"]["result"] == {}
    assert response.data["log"]["command"] == {"action": method, **arguments}
    assert response.data["step"] == 1
    assert response.data["task_language"] == _TASK_LANGUAGE
    assert response.data["success"] is (blocked == "success")
    assert response.data["robocasa_terminated"] is (blocked == "success")
    assert len(response.data["images"]) == len(response.images) == 3
    assert len(env.actions) == len(reference_env.actions)
    for actual, expected in zip(env.actions, reference_env.actions, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for record in json.loads((output / "states.json").read_text())["steps"]:
        _assert_no_planner_feedback(record)
    history = robot.execute_tool("view_env_state", {"step": 1})
    assert history.data == response.data


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
def test_dispatch_preserves_policy_history_and_manual_motion_reseeding(
    dispatcher, method: str
) -> None:
    robot, env, vla, _ = dispatcher
    arguments = {"prompt": _TASK_LANGUAGE, "max_chunks": 1, "n_action_steps": 1}

    responses = [robot.execute_tool(method, arguments) for _ in range(2)]
    responses.append(robot.execute_tool("set_gripper", {"steps": 1}))
    responses.append(robot.execute_tool(method, arguments))

    for response in responses:
        _assert_no_planner_feedback(response.data)
        assert response.data["log"]["result"] == {}
    assert [options["reset_memory"] for _, options in vla.calls] == [
        [True],
        [False],
        [True],
    ]
    assert vla.calls[1][0]["video.robot0_agentview_left"][0, :, 0, 0, 0].tolist() == [
        0,
        1,
    ]
    assert vla.calls[2][0]["video.robot0_agentview_left"][0, :, 0, 0, 0].tolist() == [
        3,
        3,
    ]
    assert len(env.actions) == 4
    assert env.reset_calls == 1
    assert vla.reset_calls == 0


def test_dispatch_filters_native_fallback_when_state_capture_fails(
    dispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot, env, _, _ = dispatcher

    def fail_capture(**kwargs):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(robot, "get_env_state", fail_capture)
    response = robot.execute_tool("move_to", {"xyz": [0.32, 0.02, 0.93]})

    assert response.is_error
    assert "camera unavailable" in response.data["error"]
    assert response.data["state_capture_error"] == "camera unavailable"
    _assert_no_planner_feedback(response.data)
    assert env.actions


def test_dispatch_preserves_validation_and_reset_errors(dispatcher) -> None:
    robot, env, _, _ = dispatcher

    invalid = robot.execute_tool("move_to", {})
    denied = robot.execute_tool("reset", {})

    assert invalid.is_error and denied.is_error
    assert invalid.data["error"] == "bad arguments for move_to"
    assert invalid.data["errors"][0]["loc"] == ("xyz",)
    assert "DISABLED" in denied.data["error"]
    assert "DISABLED" in denied.data["log"]["result"]["error"]
    _assert_no_planner_feedback(invalid.data)
    _assert_no_planner_feedback(denied.data)
    assert env.actions == []


@pytest.mark.parametrize("status", ["success", "failure", "stuck"])
def test_finish_preserves_the_agents_declared_outcome(dispatcher, status: str) -> None:
    robot, env, _, _ = dispatcher

    response = robot.execute_tool(
        "finish", {"status": status, "summary": "Agent summary."}
    )

    assert response.data == {
        "_finish": True,
        "status": status,
        "summary": "Agent summary.",
    }
    assert env.actions == []


@pytest.mark.parametrize("profile", ["hf", "local"])
def test_valid_corpus_is_preserved_but_not_exposed(tmp_path, make_corpus, profile):
    root = make_corpus(tmp_path / "corpus")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    selection = TaskMemory.load(root, "OpenDrawer", profile=profile)

    assert selection.contents == {}
    assert selection.selected == selection.missing == ()
    assert selection.task_family == {}
    assert selection.metadata() == {
        "policy": "none",
        "selected_files": [],
        "missing_layers": [],
        "profile": profile,
        "task_family": {},
    }
    assert before == {p: p.read_bytes() for p in before}


def test_file_tools_deny_memory_source_and_state_reads_including_aliases(
    tmp_path, monkeypatch, make_corpus
):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    root = make_corpus(tmp_path / "corpus")
    output = tmp_path / "episode"
    manager = RoboCasaMemoryManager(
        TaskMemory.load(root, "OpenDrawer"), output_dir=output
    )
    files = CommonTools(memory=manager)
    corpus_alias = tmp_path / "corpus-alias"
    corpus_alias.symlink_to(root, target_is_directory=True)
    historical = root / "global/GLOBAL_MEMORY.md"
    file_alias = tmp_path / "historical.md"
    file_alias.symlink_to(historical)
    targets = [historical, file_alias, historical.relative_to(tmp_path)]
    for leaf in root.rglob("*"):
        if leaf.is_file():
            targets.extend([leaf, corpus_alias / leaf.relative_to(root)])
    for name in ("states.json", "success_criteria.md", "simulator_source.py"):
        outside = output / name
        outside.write_text("forbidden simulator information")
        escape = root / name
        escape.symlink_to(outside)
        targets.extend([outside, escape, root / ".." / "episode" / name])
    for target in targets:
        with pytest.raises(PermissionError):
            files.read_text_file(path=str(target))
    with pytest.raises(PermissionError):
        files.write_text_file(path=str(historical), content="must not overwrite")
    assert manager.unread_files == ()
    assert (output / "memory_reads.jsonl").read_text() == ""
    assert files.list_dir(path=str(root)).data["files"] == []

    audit = output / "agent_audit.json"
    written = files.write_text_file(
        path=str(audit), content='{"summary":"this episode"}'
    )
    assert not written.is_error
    assert json.loads(audit.read_text()) == {"summary": "this episode"}
    with pytest.raises(PermissionError):
        files.read_text_file(path=str(audit))


def test_rendered_evaluation_has_no_unavailable_inputs(tmp_path, make_corpus):
    root = make_corpus(tmp_path / "corpus")
    prompt = get_robot_spec().prompts.render(
        "system",
        variables={
            "task_name": "OpenDrawer",
            "split": "target",
            "memory_profile": "local",
            "memory_dir": str(root),
        },
    )
    assert "No historical task memory" in prompt
    assert "complete live task_language" in prompt
    assert "state.success" in prompt
    assert str(root) not in prompt
    assert "Available memory files" not in prompt
    assert "{{" not in prompt
    for forbidden in (
        "back_project_batch",
        "query_world_map",
        "get_scene_graph",
        "task_progress",
        "success_criteria",
        "grasp_contact",
        "grasp_obj",
    ):
        assert forbidden not in prompt


def test_real_tool_dispatch_retains_success_without_removed_feedback(
    execution, tmp_path, monkeypatch, make_corpus
):
    _, env, vla = execution
    root = make_corpus(tmp_path / "corpus")
    output = tmp_path / "toolkit"
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": str(output)}
    )
    manager = RoboCasaMemoryManager(
        TaskMemory.load(root, "OpenDrawer"), output_dir=output
    )
    robot = RoboCasaToolkit(
        runtime_kwargs={
            "env_client": env,
            "vla_client": vla,
            "workdir": tmp_path / "toolkit-work",
            "hi_res": 0,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=manager,
        state_output_dir=output,
    )
    robot._primitives._rldx._unmap = lambda action: action
    assert not (output / "success_criteria.md").exists()
    names = {tool.name for tool in robot.list_tools()}
    for name in ("back_project_batch", "query_world_map", "get_scene_graph"):
        assert name not in names
        assert robot.execute_tool(name, {}).data == {"error": f"unknown tool: {name}"}
    assert env.actions == []
    first = robot.execute_tool("view_env_state", {"step": 0})
    assert first.data["success"] is False
    assert robot.solved() is False
    env.success_after = 2
    grip = robot.execute_tool("set_gripper", {"gripper": 1, "steps": 1})
    action = robot.execute_tool(
        "rldx_skill",
        {
            "prompt": _TASK_LANGUAGE,
            "max_chunks": 5,
            "n_action_steps": 1,
        },
    )
    assert action.data["success"] is True
    assert action.data["robocasa_terminated"] is True
    assert action.data["log"]["result"] == {}
    assert len(vla.calls) == 1
    assert len(env.actions) == 2
    assert robot.solved() is True
    history = robot.execute_tool("view_env_state", {"step": 0})
    assert history.data == first.data
    for response in (first, grip, action, history):
        assert not response.is_error
        _assert_no_planner_feedback(response.data)
        assert response.data["state"] == {}
    for record in json.loads((output / "states.json").read_text())["steps"]:
        _assert_no_planner_feedback(record)
    assert env.progress_queries == 0
    assert (output / "memory_reads.jsonl").read_text() == ""
    assert (
        robot.execute_tool(
            "finish",
            {
                "status": "success",
                "summary": "Native success was true.",
            },
        ).data["_finish"]
        is True
    )
