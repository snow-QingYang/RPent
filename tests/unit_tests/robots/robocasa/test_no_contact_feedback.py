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

"""Contact feedback removal through real VLA execution and state persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robots.robocasa import tools
from robots.robocasa.primitives import RoboCasaPrimitives
from rpent.session import EnvState

_CONTACT_FIELDS = {"grasped", "grasp_detected", "grasp_contact", "grasp_obj"}
_TASK_LANGUAGE = "Scrub the cutting board with the sponge."


class _Env:
    """CPU environment boundary with observable steps and forbidden contact RPC."""

    def __init__(self) -> None:
        self.task_name = "ScrubCuttingBoard"
        self.actions: list[np.ndarray] = []
        self.reset_calls = 0
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
        return self.progress

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


def _call(primitives: RoboCasaPrimitives, method: str, **kwargs: Any) -> dict[str, Any]:
    return getattr(primitives, method)(
        prompt=_TASK_LANGUAGE,
        max_chunks=kwargs.pop("max_chunks", 1),
        n_action_steps=1,
        settle_patience=999,
        **kwargs,
    ).data


def _assert_no_contact_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert not (_CONTACT_FIELDS & value.keys())
        for child in value.values():
            _assert_no_contact_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_contact_fields(child)


@pytest.mark.parametrize("method,base_motion", [("rldx_skill", 0.8), ("rldx_arm", 0.1)])
def test_vla_outputs_proprioception_without_querying_contact(
    execution, method: str, base_motion: float
) -> None:
    primitives, env, vla = execution

    result = _call(primitives, method)

    _assert_no_contact_fields(result)
    assert result["held_apart"] is True
    assert result["gripper_qpos"] == 0.02
    assert result["status"] == "cap"
    assert result["steps_applied"] == 1
    assert len(vla.calls) == len(env.actions) == 1
    np.testing.assert_allclose(env.actions[0][7:11], base_motion)


@pytest.mark.parametrize("method", ["rldx_skill", "rldx_arm"])
def test_native_success_still_ends_the_vla_loop(execution, method: str) -> None:
    primitives, env, vla = execution
    env.success_after = 1

    result = _call(primitives, method, max_chunks=5)

    _assert_no_contact_fields(result)
    assert result["status"] == "success"
    assert result["chunks"] == result["steps_applied"] == 1
    assert len(vla.calls) == len(env.actions) == 1


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


@pytest.mark.parametrize(
    "task,progress",
    [
        (
            "ScrubCuttingBoard",
            {"board_contact_timer": 6, "sweep_range": 0.2, "gripper_obj_far": True},
        ),
        (
            "PrepareCoffee",
            {
                "contact_check": True,
                "gripper_obj_far": True,
                "coffee_machine__turned_on": True,
            },
        ),
        (
            "MakeIceLemonade",
            {
                "lemon_in_cup": True,
                "ice1_in_cup": True,
                "ice2_in_cup": False,
                "ice_in_cup": True,
                "success": True,
                "gripper_far_lemon": True,
                "gripper_far_ice": True,
            },
        ),
        (
            "PanTransfer",
            {
                "_robot_touched_food": True,
                "vegetable_on_plate": False,
                "pan_on_stove": True,
            },
        ),
        (
            "KettleBoiling",
            {
                "obj_on_stove": True,
                "dist": 0.1,
                "kettle_on_site": True,
                "burner_on": False,
            },
        ),
        (
            "AnotherTask",
            {"grasped": True, "contact_timer": 5, "future_progress_field": 0.3},
        ),
    ],
)
def test_native_progress_is_unchanged_including_contact_derivatives(
    execution, task: str, progress: dict[str, Any]
) -> None:
    primitives, env, _ = execution
    env.task_name = task
    env.progress = progress.copy()
    env.success_after = 0

    state = primitives.current_state_dict()

    assert state["task_progress"] == progress
    assert state["success"] is True
    assert state["robocasa_terminated"] is True
    assert state["state"]["robot0_gripper_qpos"] == [0.02, 0.02]
    assert env.progress == progress


def test_saved_states_and_history_do_not_reintroduce_contact_feedback(
    execution, tmp_path: Path
) -> None:
    primitives, env, _ = execution
    run_dir = tmp_path / "run"
    state = EnvState(run_dir)
    returned = []
    for method in ("rldx_skill", "rldx_arm"):
        result = _call(primitives, method)
        returned.append(result)
        tools.dump_state(
            primitives,
            state,
            log={"command": {"name": method}, "result": result, "elapsed_s": 0.1},
        )

    # Exercise the actual latest and historical state tools, including image IO.
    for requested_step, expected_step in ((None, 1), (0, 0), (1, 1)):
        response = tools.view_env_state(step=requested_step, state=state)
        _assert_no_contact_fields(response.data["log"]["result"])
        assert response.data["step"] == expected_step
        assert response.data["log"]["result"] == returned[expected_step]
        assert response.data["task_progress"] == env.progress
        assert len(response.data["images"]) == 3

    persisted = json.loads((run_dir / "states.json").read_text())
    assert len(persisted["steps"]) == 2
    for record in persisted["steps"]:
        _assert_no_contact_fields(record["result"])
        assert record["extras"]["task_progress"] == env.progress
        assert record["result"]["held_apart"] is True
        assert record["result"]["gripper_qpos"] == 0.02
