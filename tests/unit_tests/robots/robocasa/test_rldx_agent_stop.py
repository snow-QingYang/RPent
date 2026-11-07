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

"""RLDX action execution stays independent of the native task success verdict."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robots.robocasa.rldx_skill import RLDXSkill

_TASK = "Put the bowl in the cabinet."


class _Env:
    def __init__(self, *, success: bool = True, moving: bool = True) -> None:
        self.success = success
        self.moving = moving
        self.success_checks = 0
        self.contact_checks = 0
        self.actions: list[np.ndarray] = []
        self.eef_pos = np.array([0.2, 0.0, 0.9])
        self.gripper_qpos = np.array([0.02, 0.02])
        self.current_raw_obs = {
            "robot0_gripper_qpos": self.gripper_qpos,
            "robot0_base_pos": np.zeros(3),
            "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_base_to_eef_pos": self.eef_pos,
            "robot0_base_to_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }

    def step(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())
        if self.moving:
            self.eef_pos[0] += 0.02

    def check_success(self) -> bool:
        self.success_checks += 1
        return self.success

    def grasp_contact(self) -> tuple[bool, str]:
        self.contact_checks += 1
        return True, "bowl"

    def render_camera(self, camera: str, *, height: int, width: int) -> np.ndarray:
        return np.full((4, 4, 3), len(self.actions), dtype=np.uint8)

    def reassemble_env_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(list(action.values()))


class _Vla:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

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


@pytest.fixture
def execution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RLDX_VIDEO_DIR", raising=False)
    env, vla = _Env(), _Vla()
    skill = RLDXSkill(env, vla)
    # Replace only the optional simulator conversion; run the real execution,
    # observation, contact, history and stopping logic on CPU.
    skill._unmap = lambda action: action
    return skill, env, vla


@pytest.mark.parametrize("base_clip,expected_base", [(None, 0.8), (0.1, 0.1)])
def test_native_success_does_not_shorten_vla_action_budget(
    execution, base_clip: float | None, expected_base: float
) -> None:
    skill, env, vla = execution

    result = skill.run(
        _TASK, max_chunks=3, n_action_steps=2, settle_patience=999, base_clip=base_clip
    )

    assert result["status"] == "cap"
    assert result["chunks"] == len(vla.calls) == env.success_checks == 3
    assert result["steps_applied"] == len(env.actions) == 6
    assert "success" not in result
    assert env.contact_checks == 4  # Every chunk and the final grasp report.
    assert result["grasp_contact"] is True
    assert result["grasp_obj"] == "bowl"
    assert result["grasped"] is result["grasp_detected"] is True
    for action in env.actions:
        np.testing.assert_allclose(action[7:11], expected_base)
    assert [options["reset_memory"] for _, options in vla.calls] == [
        [True],
        [False],
        [False],
    ]


@pytest.mark.parametrize("native_success", [True, False])
def test_motion_settling_is_independent_of_native_success(
    execution, native_success: bool
) -> None:
    skill, env, vla = execution
    env.success = native_success
    env.moving = False

    result = skill.run(_TASK, max_chunks=5, n_action_steps=2, settle_patience=2)

    assert result["status"] == "settled"
    assert result["chunks"] == len(vla.calls) == env.success_checks == 2
    assert result["steps_applied"] == len(env.actions) == 4
    assert "success" not in result


def test_same_task_after_native_success_preserves_vla_history(execution) -> None:
    skill, env, vla = execution

    first = skill.run(_TASK, max_chunks=1, n_action_steps=2, settle_patience=999)
    second = skill.run(_TASK, max_chunks=1, n_action_steps=2, settle_patience=999)

    assert first["status"] == second["status"] == "cap"
    assert len(env.actions) == 4
    assert env.success_checks == 2
    assert [options["reset_memory"] for _, options in vla.calls] == [[True], [False]]
    second_video = vla.calls[1][0]["video.robot0_agentview_left"]
    assert second_video[0, :, 0, 0, 0].tolist() == [0, 2]
