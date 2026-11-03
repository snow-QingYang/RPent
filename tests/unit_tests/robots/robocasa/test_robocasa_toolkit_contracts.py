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

"""Offline contracts for the RoboCasa toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.robocasa import robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig
from rpent.tools import Toolkit, ToolResult
from rpent.tools.common import CommonTools
from rpent.utils import templates

COMMON_TOOLS = {"read_text_file", "write_text_file", "list_dir", "finish"}

EXPECTED_TOOLS = COMMON_TOOLS | {
    "move_to",
    "move_delta",
    "rotate_pitch",
    "set_gripper",
    "release",
    "scripted_grasp",
    "rldx_skill",
    "rldx_arm",
    "navigate_to",
    "move_base",
    "reset",
    "view_env_state",
    "back_project_batch",
    "query_world_map",
}


def _record(step_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(step_idx=step_idx, terminated=False, extras={})


def _tool_names(robot_toolkit: Toolkit) -> set[str]:
    return {definition.name for definition in robot_toolkit.list_tools()}


def _readonly_names(robot_toolkit: Toolkit) -> set[str]:
    return {
        definition.name
        for definition in robot_toolkit.list_tools()
        if definition.readonly
    }


def test_toolkit_falls_back_to_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_corpus,
) -> None:
    memory_dir = tmp_path / "robocasa"
    make_corpus(memory_dir)
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(robot_spec, "get_memory_dir", lambda _: memory_dir)
    monkeypatch.setattr(
        toolkit,
        "RoboCasaToolkit",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={"task_name": "OpenDrawer"},
    )

    robot_toolkit = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert robot_toolkit.memory.root == memory_dir.resolve()


@pytest.mark.parametrize("mode", ["evaluation", "exploration"])
def test_toolkit_constructs_and_classifies_tools_with_a_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
    mode: str,
) -> None:
    monkeypatch.delenv("RLDX_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("RLDX_SETTLE_PATIENCE", raising=False)
    import robots.robocasa.primitives as primitives_module

    fake_single_arm_primitives = fake_single_arm_primitives.for_robot(
        primitives_module.RoboCasaPrimitives
    )
    dumped: list[Any] = []
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )
    monkeypatch.setattr(
        primitives_module,
        "RoboCasaPrimitives",
        fake_single_arm_primitives,
    )
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        toolkit.robocasa_tools,
        "dump_state",
        lambda primitives, state, log: dumped.append(primitives) or _record(),
    )

    robot_toolkit = toolkit.RoboCasaToolkit(
        runtime_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        mode=mode,
    )

    assert _tool_names(robot_toolkit) == EXPECTED_TOOLS
    assert _readonly_names(robot_toolkit) == COMMON_TOOLS | {
        "view_env_state",
        "back_project_batch",
        "query_world_map",
    }
    assert dumped == fake_single_arm_primitives.instances
    primitive = fake_single_arm_primitives.instances[0]
    assert primitive.reset_calls == (1 if mode == "evaluation" else 0)
    assert primitive.kwargs["allow_reset"] is (mode == "exploration")
    assert primitive.recording_started is True
    assert callable(primitive.kwargs["check_cancelled"])
    assert (tmp_path / "success_criteria.md").read_text() == "offline success criteria"


def test_solved_reads_only_the_final_environment_record() -> None:
    records = [SimpleNamespace(extras={"success": True})]
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    robot_toolkit._state = SimpleNamespace(latest_record=lambda: records[-1])

    assert robot_toolkit.solved() is True

    records.append(SimpleNamespace(extras={"success": False}))
    assert robot_toolkit.solved() is False


def test_observations_hide_task_progress_but_preserve_internal_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_single_arm_primitives: type[Any],
) -> None:
    import robots.robocasa.primitives as primitives_module

    fake_primitives = fake_single_arm_primitives.for_robot(
        primitives_module.RoboCasaPrimitives
    )
    snapshots = [
        {
            "task_language": "Open the drawer.",
            "state": {"robot0_base_pos": [0.0, 0.0, 0.0]},
            "success": False,
            "robocasa_terminated": False,
            "task_progress": {"drawer_open": False, "dist": 0.5},
        },
        {
            "task_language": "Open the drawer.",
            "state": {"robot0_base_pos": [0.1, 0.0, 0.0]},
            "success": True,
            "robocasa_terminated": True,
            "task_progress": {"drawer_open": True, "dist": 0.0},
        },
    ]
    next_snapshot = iter(snapshots)
    monkeypatch.setattr(
        fake_primitives,
        "current_state_dict",
        lambda self: next(next_snapshot),
        raising=False,
    )
    monkeypatch.setattr(fake_primitives, "_vla_desync", False, raising=False)
    monkeypatch.setattr(fake_primitives, "_keep_heavy", 2, raising=False)
    monkeypatch.setattr(primitives_module, "RoboCasaPrimitives", fake_primitives)
    monkeypatch.setattr(
        toolkit.robocasa_tools,
        "_save_observation_artifacts",
        lambda primitives, state, step_idx: None,
    )
    output_dir = tmp_path / "run"
    robot_toolkit = toolkit.RoboCasaToolkit(
        runtime_kwargs={"env_client": object(), "vla_client": object()},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=output_dir,
    )

    initial = robot_toolkit.execute_tool("view_env_state", {"step": 0})
    assert robot_toolkit.solved() is False
    action = robot_toolkit.execute_tool("move_base", {"forward": 0.1, "steps": 1})
    history = robot_toolkit.execute_tool("view_env_state", {"step": 0})
    assert history.data == initial.data
    assert robot_toolkit.solved() is True

    saved = json.loads((output_dir / "states.json").read_text())["steps"]
    assert len(saved) == 2
    for step, (response, expected) in enumerate(zip((initial, action), snapshots)):
        assert not response.is_error
        assert "task_progress" not in response.data
        assert response.data["state"] == expected["state"]
        assert response.data["success"] == expected["success"]
        assert response.data["robocasa_terminated"] == expected["robocasa_terminated"]
        assert response.data["task_language"] == expected["task_language"]
        record = robot_toolkit.state.get(step)
        assert record.extras["task_progress"] == expected["task_progress"]
        assert record.state == expected["state"]
        assert record.terminated == expected["robocasa_terminated"]
        assert saved[step]["extras"]["task_progress"] == expected["task_progress"]
        assert saved[step]["extras"]["success"] == expected["success"]


@pytest.mark.parametrize("reading", ["none", "partial", "complete"])
@pytest.mark.parametrize("allow_reset", [False, True])
def test_optional_reading_preserves_no_reset(
    tmp_path, monkeypatch, make_corpus, reading, allow_reset
):
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
    from robots.robocasa.primitives import RoboCasaPrimitives
    from rpent.session import EnvState

    root = make_corpus(tmp_path / "robocasa")
    memory = RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"))
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    Toolkit.__init__(
        robot_toolkit,
        dashboard_events=NullDashboardEventSink(),
        memory=memory,
        state=EnvState(tmp_path / "run"),
    )
    resets = []
    primitive = RoboCasaPrimitives.__new__(RoboCasaPrimitives)
    primitive._allow_reset = allow_reset
    primitive.env = SimpleNamespace(
        reset=lambda: resets.append("env"),
        eef_pos=SimpleNamespace(tolist=lambda: [0, 0, 0]),
    )
    primitive._rldx = SimpleNamespace(reset_session=lambda: resets.append("vla"))
    robot_toolkit.add_tool(primitive.reset)
    monkeypatch.setattr(
        robot_toolkit,
        "get_env_state",
        lambda **kwargs: ToolResult(data=kwargs["result"]),
    )

    read = CommonTools(memory=memory).read_text_file
    if reading == "partial":
        read(path=str(root / memory.selection.selected[0]), max_chars=1)
    elif reading == "complete":
        for name in memory.selection.selected:
            read(path=str(root / name))
    result = robot_toolkit.execute_tool("reset", {}).data
    if allow_reset:
        assert result["reset"] is True
        assert resets == ["env", "vla"]
    else:
        assert "reset is DISABLED" in result["error"]
        assert resets == []


def test_exploration_attempt_budget_and_solved_guards():
    robot_toolkit = toolkit.RoboCasaToolkit.__new__(toolkit.RoboCasaToolkit)
    robot_toolkit._attempt = 1
    robot_toolkit._attempts_per_session = 2
    records = [SimpleNamespace(extras={"success": False})]
    robot_toolkit._state = SimpleNamespace(latest_record=lambda: records[-1])
    resets = []
    robot_toolkit._primitives = SimpleNamespace(
        reset=lambda: resets.append(1) or ToolResult()
    )

    def inner(**kwargs):
        return ToolResult(data=kwargs)

    assert (
        robot_toolkit._guarded_finish(inner, status="stuck").data["error"]
        == "finish refused"
    )
    assert robot_toolkit._reset_episode().data["attempt"] == 2
    assert robot_toolkit._reset_episode().data["error"] == "reset refused"
    assert resets == [1]
    assert robot_toolkit._guarded_finish(inner, status="stuck").data == {
        "status": "stuck"
    }
    records.append(SimpleNamespace(extras={"success": True}))
    assert robot_toolkit._reset_episode().data["error"] == "reset refused"
    assert robot_toolkit._guarded_finish(inner, status="success").data == {
        "status": "success"
    }


def test_exploration_recipe_exports_only_winning_attempt(tmp_path):
    import json

    from robots.robocasa.tools import write_recipe_from_states

    commands = [
        ("move_to", {}),
        ("reset", {}),
        ("move_delta", {}),
        ("reset", {"error": "refused"}),
        ("move_base", {"error": "failed"}),
        ("finish", {}),
    ]
    records = [
        SimpleNamespace(
            step_idx=index,
            command={"action": action},
            result=result,
            extras={"success": False},
        )
        for index, (action, result) in enumerate(commands)
    ]
    state = SimpleNamespace(records=lambda: records)
    assert (
        write_recipe_from_states(state, "OpenDrawer_target_s0", output_dir=tmp_path)
        == ""
    )
    records[-1].extras["success"] = True
    name = write_recipe_from_states(state, "OpenDrawer_target_s0", output_dir=tmp_path)
    assert [
        json.loads(line) for line in (tmp_path / name).read_text().splitlines()
    ] == [{"action": "move_delta"}]
