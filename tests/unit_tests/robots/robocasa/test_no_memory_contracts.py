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

"""Offline contracts for evaluation without historical task memory."""

from __future__ import annotations

import json

import pytest

from robots.robocasa.eval.result import finalize_cell_result
from robots.robocasa.eval.validate_target50 import DEFAULT_MANIFEST, validate_results
from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
from robots.robocasa.robot_spec import get_robot_spec
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.tools.common import CommonTools


@pytest.mark.parametrize("profile", ["hf", "local"])
def test_existing_corpus_is_preserved_but_exposes_no_memory(
    tmp_path, make_corpus, profile
):
    root = make_corpus(tmp_path / "corpus")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert any(path.parent.name == "task-specific" for path in before)
    assert any(path.parent.name == "global" for path in before)

    selection = TaskMemory.load(root, "OpenDrawer", profile=profile)

    assert selection.contents == {}
    assert selection.selected == ()
    assert selection.missing == ()
    assert selection.task_family == {}
    assert selection.profile == profile
    assert selection.metadata()["policy"] == "none"
    assert before == {path: path.read_bytes() for path in before}


def test_local_task_family_is_not_exposed(tmp_path, make_native_corpus):
    root = make_native_corpus(tmp_path / "corpus")
    assert list((root / "task-family").glob("*.md"))
    selection = TaskMemory.load(root, "OpenDrawer", profile="local")
    assert selection.contents == {}
    assert selection.selected == ()
    assert selection.task_family == {}


def test_file_tools_block_corpus_access_through_direct_and_alias_paths(
    tmp_path, monkeypatch, make_corpus
):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    root = make_corpus(tmp_path / "corpus")
    files = CommonTools(
        memory=RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"))
    )
    alias = tmp_path / "corpus-alias"
    alias.symlink_to(root, target_is_directory=True)
    target = root / "global/GLOBAL_MEMORY.md"
    file_alias = tmp_path / "historical-note.md"
    file_alias.symlink_to(target)
    original = target.read_text()

    for path in (root, alias, alias.resolve()):
        assert files.list_dir(path=str(path)).data["files"] == []
        for directory in ("global", "task-specific"):
            with pytest.raises(PermissionError):
                files.list_dir(path=str(path / directory))
        for leaf in root.rglob("*"):
            if leaf.is_file():
                with pytest.raises(PermissionError):
                    files.read_text_file(path=str(path / leaf.relative_to(root)))

    for path in (file_alias, file_alias.resolve(), target.relative_to(tmp_path)):
        with pytest.raises(PermissionError):
            files.read_text_file(path=str(path))
        with pytest.raises(PermissionError):
            files.write_text_file(path=str(path), content="must not overwrite memory")
    assert target.read_text() == original


def test_current_observations_and_success_rules_remain_readable(tmp_path, make_corpus):
    root = make_corpus(tmp_path / "corpus")
    output = tmp_path / "episode"
    manager = RoboCasaMemoryManager(
        TaskMemory.load(root, "OpenDrawer"), output_dir=output
    )
    files = CommonTools(memory=manager)
    current = {
        "state_00.json": '{"success": false, "task_progress": {"dist": 0.2}}',
        "success_criteria.md": "def _check_success(self): return self.door_open",
    }
    for name, content in current.items():
        path = output / name
        path.write_text(content)
        assert files.read_text_file(path=str(path)).data["content"] == content
    assert set(current).issubset(files.list_dir(path=str(output)).data["files"])
    assert (output / "memory_reads.jsonl").read_text() == ""


def test_prompt_omits_corpus_paths_and_retains_baseline_observation_rules(
    tmp_path, make_corpus
):
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
    assert str(root) not in prompt
    assert "Available memory files" not in prompt
    assert "task-specific/" not in prompt
    assert "{{" not in prompt
    for instruction in (
        "state.success",
        "task_progress",
        "back_project_batch",
        "query_world_map",
    ):
        assert instruction in prompt


@pytest.fixture
def finalized_no_memory_case(tmp_path, monkeypatch, make_corpus):
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    atomic = manifest["splits"]["atomic"]
    manifest["splits"] = {
        "atomic": {
            **atomic,
            "tasks": ["OpenDrawer"],
            "task_count": 1,
            "seeds": [1],
            "cell_count": 1,
        }
    }
    manifest["total_cells"] = manifest["total_tasks"] = 1
    manifest_path = write_json_atomic(tmp_path / "manifest.json", manifest)
    runtime = manifest["runtime_protocol"]
    for variable, field in (
        ("RLDX_MAX_CHUNKS", "rldx_max_chunks"),
        ("RLDX_SETTLE_PATIENCE", "rldx_settle_patience"),
        ("RLDX_ACTION_STEPS_PER_CHUNK", "rldx_action_steps_per_chunk"),
    ):
        monkeypatch.setenv(variable, str(runtime[field]))
    output = tmp_path / "results/atomic/OpenDrawer_s1"
    root = make_corpus(tmp_path / "corpus")
    RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"), output_dir=output)
    reference = manifest["planner_reference"]
    result_path = finalize_cell_result(
        RunFinalizationContext(
            output_dir=output,
            robot_name="robocasa",
            task_desc={"task_name": "OpenDrawer", "split": "target", "seed": 1},
            environment_success=True,
            agent_error=None,
            elapsed_s=1.0,
            planner=reference["planner"],
            model=reference["model"],
            reasoning_effort=reference["reasoning_effort"],
            max_turns=reference["max_turns"],
            planner_timeout_s=atomic["timeout_seconds"],
            finish_result=None,
            stats={},
        )
    )
    return tmp_path / "results", manifest_path, result_path


def test_finalized_result_has_valid_empty_memory_audit(finalized_no_memory_case):
    results, manifest, path = finalized_no_memory_case
    result = json.loads(path.read_text())
    assert result["success"] is True
    assert result["valid"] is True
    assert result["memory"]["policy"] == "none"
    assert result["memory"]["selected_files"] == []
    assert result["memory"]["read_files"] == []
    assert result["memory"]["audit_status"] == "ok"
    summary, errors = validate_results(results, manifest_path=manifest)
    assert errors == []
    assert summary["valid_cells"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_files", ["global/GLOBAL_MEMORY.md"], "empty memory selection"),
        ("read_files", ["global/GLOBAL_MEMORY.md"], "empty memory read_files"),
        ("policy", "task-global", "memory policy"),
    ],
)
def test_validator_rejects_memory_in_an_otherwise_valid_case(
    finalized_no_memory_case, field, value, message
):
    results, manifest, path = finalized_no_memory_case
    summary, errors = validate_results(results, manifest_path=manifest)
    assert errors == []
    assert summary["valid_cells"] == 1
    result = json.loads(path.read_text())
    result["memory"][field] = value
    write_json_atomic(path, result)

    summary, errors = validate_results(results, manifest_path=manifest)
    assert summary["valid_cells"] == 0
    assert any(message in error for error in errors)
