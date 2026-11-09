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

_HIDDEN_FIELDS = {
    "task_progress",
    "grasped",
    "grasp_detected",
    "grasp_contact",
    "grasp_obj",
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


def _assert_no_hidden_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert not (_HIDDEN_FIELDS & value.keys())
        for child in value.values():
            _assert_no_hidden_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_hidden_fields(child)


@pytest.mark.parametrize("method,base_motion", [("rldx_skill", 0.8), ("rldx_arm", 0.1)])
def test_vla_outputs_proprioception_without_querying_contact(
    execution, method: str, base_motion: float
) -> None:
    primitives, env, vla = execution

    result = _call(primitives, method)

    _assert_no_hidden_fields(result)
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

    _assert_no_hidden_fields(result)
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


@pytest.mark.parametrize("success", [False, True])
def test_current_state_omits_progress_without_calling_progress_rpc(execution, success):
    primitives, env, _ = execution
    env.success_after = 0 if success else None

    state = primitives.current_state_dict()

    _assert_no_hidden_fields(state)
    assert env.progress_queries == 0
    assert state["success"] is success
    assert state["robocasa_terminated"] is success
    assert state["state"]["robot0_gripper_qpos"] == [0.02, 0.02]
    assert state["state"]["robot0_base_pos"] == [0.0, 0.0, 0.0]


def test_saved_states_and_history_do_not_reintroduce_contact_feedback(
    execution, tmp_path: Path
) -> None:
    primitives, _, _ = execution
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
        _assert_no_hidden_fields(response.data)
        assert response.data["step"] == expected_step
        assert response.data["log"]["result"] == returned[expected_step]
        assert len(response.data["images"]) == 3

    persisted = json.loads((run_dir / "states.json").read_text())
    _assert_no_hidden_fields(persisted)
    assert len(persisted["steps"]) == 2
    for record in persisted["steps"]:
        assert record["result"]["held_apart"] is True
        assert record["result"]["gripper_qpos"] == 0.02


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
    assert action.data["log"]["result"]["status"] == "success"
    assert action.data["log"]["result"]["steps_applied"] == 1
    assert robot.solved() is True
    history = robot.execute_tool("view_env_state", {"step": 0})
    assert history.data == first.data
    for response in (first, grip, action, history):
        assert not response.is_error
        _assert_no_hidden_fields(response.data)
    _assert_no_hidden_fields(json.loads((output / "states.json").read_text()))
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
