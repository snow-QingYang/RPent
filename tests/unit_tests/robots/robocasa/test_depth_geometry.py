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

"""Offline geometry contracts; prediction is synthetic, never a model download."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa import depth_geometry, tools
from rpent.session import EnvState


def _camera_metadata(shape, *, intrinsic=None, cam2world=None):
    return {
        "height": shape[0],
        "width": shape[1],
        "intrinsic": np.asarray(np.eye(3) if intrinsic is None else intrinsic).tolist(),
        "extrinsic_cam2world": np.asarray(
            np.eye(4) if cam2world is None else cam2world
        ).tolist(),
    }


def _save_camera(state, prefix, *, shape=(8, 8), value=0, **calibration):
    state.save(f"{prefix}.png", np.full((*shape, 3), value, dtype=np.uint8))
    state.save(f"{prefix}_metadata.json", _camera_metadata(shape, **calibration))


@pytest.fixture
def recorded_state(tmp_path: Path) -> EnvState:
    state = EnvState(tmp_path / "run")
    with state.record_step(state={}):
        _save_camera(state, "agentview")
        # Existing GT artifacts must not be used, even if prediction fails.
        state.save("agentview_world.npz", np.full((8, 8, 3), 999.0))
        state.save("agentview_depth.npz", np.full((8, 8), 999.0))
    return state


@pytest.fixture
def synthetic_predictor(monkeypatch):
    calls = []

    def predict(rgb, intrinsic, *, device=None):
        calls.append((rgb.copy(), intrinsic.copy()))
        return np.ones(rgb.shape[:2], dtype=np.float32)

    monkeypatch.setattr(depth_geometry, "_predict_metric_depth", predict)
    return calls


def _reject_gt_reads(monkeypatch, state):
    original = state.load

    def load(name, *, step=-1):
        assert "_world" not in name, f"read simulator world map: {name}"
        assert not name.endswith("_depth.npz"), f"read simulator depth: {name}"
        return original(name, step=step)

    monkeypatch.setattr(state, "load", load)


def test_projection_uses_row_column_focal_principal_point_and_camera_pose():
    depth = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)
    intrinsic = np.array([[2, 0, 1], [0, 4, 0.5], [0, 0, 1]])
    cam2world = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
    expected = np.array(
        [
            [[10.25, 19, 32], [10.5, 20, 34], [10.75, 23, 36]],
            [[9, 16, 38], [8.75, 20, 40], [8.5, 26, 42]],
        ]
    )

    geometry = depth_geometry.DepthGeometry(depth, intrinsic, cam2world)
    np.testing.assert_allclose(
        depth_geometry.depth_to_world(depth, intrinsic, cam2world), expected
    )
    np.testing.assert_allclose(geometry.world_map(), expected)
    np.testing.assert_allclose(geometry.xyz_at(0, 2), expected[0, 2])
    np.testing.assert_allclose(geometry.xyz_at(1, 0), expected[1, 0])


def test_nonpositive_and_nonfinite_depths_never_produce_world_points():
    depth = np.array([[0, -1, np.nan], [np.inf, -np.inf, 2]], dtype=np.float32)
    geometry = depth_geometry.DepthGeometry(depth, np.eye(3), np.eye(4))
    world = geometry.world_map()

    assert np.isnan(world.reshape(-1, 3)[:5]).all()
    np.testing.assert_allclose(world[1, 2], [4, 2, 2])
    assert np.isnan(geometry.xyz_at(0, 0)).all()
    assert np.isnan(geometry.xyz_at(1, 0)).all()


def test_metric_depth_uses_processed_focal_before_restoring_rgb_resolution():
    # Original 4x8 -> processed 2x2: fx=1200 -> 300, fy=600 -> 300.
    # A canonical prediction of 6 therefore means 6 metres, not 18 metres.
    intrinsic = np.diag([1200.0, 600.0, 1.0])
    result = depth_geometry._metric_depth_from_prediction(
        np.full((2, 2), 6.0), intrinsic, (4, 8, 3)
    )

    assert result.shape == (4, 8)
    np.testing.assert_allclose(result, 6.0)


def test_resize_keeps_invalid_depth_neighborhood_invalid():
    result = depth_geometry._metric_depth_from_prediction(
        np.array([[np.nan, 2], [2, 2]], dtype=np.float32),
        np.diag([600.0, 600.0, 1.0]),
        (4, 4, 3),
    )

    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 1])
    assert result[-1, -1] == pytest.approx(2.0)


def test_both_tools_share_prediction_cache_and_ignore_old_gt_artifacts(
    recorded_state, synthetic_predictor, monkeypatch
):
    _reject_gt_reads(monkeypatch, recorded_state)

    points = tools.back_project_batch([[1, 2]], state=recorded_state).data
    regions = tools.query_world_map(
        z_min=0.9, z_max=1.1, min_cluster_size=1, state=recorded_state
    ).data
    repeated = tools.back_project_batch([[2, 3]], state=recorded_state).data

    assert "error" not in points
    assert "error" not in regions
    assert points["results"][0]["world_xyz"] == [2.0, 1.0, 1.0]
    assert regions["clusters"]
    assert repeated["results"][0]["world_xyz"] == [3.0, 2.0, 1.0]
    assert len(synthetic_predictor) == 1
    assert recorded_state.exists("agentview_depth_predicted.npz", step=0)
    assert recorded_state.exists("agentview_depth_prediction.json", step=0)


@pytest.mark.parametrize(
    "missing", ["agentview_depth_predicted.npz", "agentview_depth_prediction.json"]
)
def test_missing_cache_artifact_recomputes_prediction(
    recorded_state, synthetic_predictor, missing
):
    depth_geometry.load_geometry(
        recorded_state, step=0, camera="agentview", resolution="low"
    )
    recorded_state.artifact_path(missing, step=0).unlink()

    geometry = depth_geometry.load_geometry(
        recorded_state, step=0, camera="agentview", resolution="low"
    )

    assert len(synthetic_predictor) == 2
    np.testing.assert_allclose(geometry.xyz_at(1, 1), [1, 1, 1])


def test_model_configuration_change_invalidates_cached_prediction(
    recorded_state, synthetic_predictor, monkeypatch, tmp_path
):
    monkeypatch.setenv("RPENT_DEPTH_MODEL_PATH", str(tmp_path / "model-a"))
    depth_geometry.load_geometry(
        recorded_state, step=0, camera="agentview", resolution="low"
    )
    monkeypatch.setenv("RPENT_DEPTH_MODEL_PATH", str(tmp_path / "model-b"))
    depth_geometry.load_geometry(
        recorded_state, step=0, camera="agentview", resolution="low"
    )

    assert len(synthetic_predictor) == 2


@pytest.mark.parametrize("configured_device", [None, "cuda:7"])
def test_prediction_receives_the_runs_configured_device(
    recorded_state, monkeypatch, configured_device
):
    devices = []
    if configured_device is not None:
        recorded_state.save(
            "depth_geometry_config.json", {"device": configured_device}, step=None
        )

    def predict(rgb, intrinsic, *, device=None):
        devices.append(device)
        return np.ones(rgb.shape[:2], dtype=np.float32)

    monkeypatch.setattr(depth_geometry, "_predict_metric_depth", predict)
    depth_geometry.load_geometry(
        recorded_state, step=0, camera="agentview", resolution="low"
    )

    assert devices == [configured_device]


@pytest.mark.parametrize(
    ("environment_device", "run_device", "expected_device"),
    [
        ("cuda:3", "cuda:7", "cuda:3"),
        (None, "cuda:7", "cuda:7"),
        (None, None, "cuda:0"),
    ],
)
def test_depth_model_device_override_precedes_run_device_and_default(
    monkeypatch, environment_device, run_device, expected_device
):
    loaded_devices = []
    if environment_device is None:
        monkeypatch.delenv("RPENT_DEPTH_DEVICE", raising=False)
    else:
        monkeypatch.setenv("RPENT_DEPTH_DEVICE", environment_device)

    class FakeModel:
        def inference(self, **kwargs):
            return SimpleNamespace(depth=np.ones((1, 2, 2), dtype=np.float32))

    def load_model(model_path, device):
        loaded_devices.append(device)
        return FakeModel()

    monkeypatch.setattr(depth_geometry, "_load_model", load_model)
    depth = depth_geometry._predict_metric_depth(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.diag([300.0, 300.0, 1.0]),
        device=run_device,
    )

    assert loaded_devices == [expected_device]
    np.testing.assert_allclose(depth, 1.0)


def test_history_queries_use_the_saved_moving_camera_pose(
    tmp_path, synthetic_predictor
):
    state = EnvState(tmp_path / "moving-camera")
    for offset in (2.0, 9.0):
        pose = np.eye(4)
        pose[0, 3] = offset
        with state.record_step(state={}):
            _save_camera(state, "wrist", cam2world=pose)

    old = depth_geometry.load_geometry(state, step=0, camera="wrist", resolution="low")
    current = depth_geometry.load_geometry(
        state, step=1, camera="wrist", resolution="low"
    )
    old_again = depth_geometry.load_geometry(
        state, step=0, camera="wrist", resolution="low"
    )

    np.testing.assert_allclose(old.xyz_at(0, 0), [2, 0, 1])
    np.testing.assert_allclose(current.xyz_at(0, 0), [9, 0, 1])
    np.testing.assert_allclose(old_again.xyz_at(0, 0), [2, 0, 1])
    assert len(synthetic_predictor) == 2


def test_high_resolution_uses_its_own_rgb_calibration_and_cache(
    tmp_path, synthetic_predictor
):
    state = EnvState(tmp_path / "resolutions")
    with state.record_step(state={}):
        _save_camera(state, "agentview", shape=(4, 4))
        _save_camera(
            state,
            "agentview_high",
            shape=(8, 8),
            value=127,
            intrinsic=np.diag([2.0, 2.0, 1.0]),
        )

    low = tools.back_project_batch([[1, 1]], state=state).data
    high = tools.back_project_batch([[2, 2]], resolution="high", state=state).data

    assert low["results"][0]["world_xyz"] == [1.0, 1.0, 1.0]
    assert high["results"][0]["world_xyz"] == [1.0, 1.0, 1.0]
    assert [rgb.shape for rgb, _ in synthetic_predictor] == [(4, 4, 3), (8, 8, 3)]
    assert synthetic_predictor[1][0][0, 0, 0] == 127
    assert state.exists("agentview_high_depth_predicted.npz", step=0)


def test_invalid_saved_calibration_fails_before_prediction(
    recorded_state, synthetic_predictor
):
    recorded_state.save("agentview_metadata.json", _camera_metadata((7, 8)), step=0)

    with pytest.raises(ValueError, match="dimensions"):
        depth_geometry.load_geometry(
            recorded_state, step=0, camera="agentview", resolution="low"
        )

    assert not synthetic_predictor


def test_prediction_failure_is_reported_without_gt_fallback(
    recorded_state, monkeypatch
):
    _reject_gt_reads(monkeypatch, recorded_state)

    def fail_prediction(rgb, intrinsic, *, device=None):
        raise RuntimeError("synthetic DA3 failure")

    monkeypatch.setattr(depth_geometry, "_predict_metric_depth", fail_prediction)
    points = tools.back_project_batch([[1, 2]], state=recorded_state).data
    regions = tools.query_world_map(state=recorded_state).data

    assert "error" in points
    assert "error" in regions
    assert not points.get("results")
    assert not regions.get("clusters")
    assert not recorded_state.exists("agentview_depth_predicted.npz", step=0)


def test_capture_uses_rgb_only_and_refreshes_all_camera_poses(tmp_path):
    class RGBCameras:
        def __init__(self):
            self.frame = 0
            self.metadata_calls = []

        def render_camera(self, camera, height=None, width=None, depth=False):
            assert depth is False, "observation capture requested simulator depth"
            return np.full((height or 4, width or 4, 3), self.frame, np.uint8)

        def get_camera_meta(self, camera, height=None, width=None):
            self.metadata_calls.append((self.frame, camera, height, width))
            pose = np.eye(4)
            pose[0, 3] = self.frame
            return _camera_metadata((height or 4, width or 4), cam2world=pose)

        def world_map(self, *args, **kwargs):
            pytest.fail("observation capture requested a simulator world map")

    env = RGBCameras()
    primitives = SimpleNamespace(env=env, hi_res=8, _cam_meta_cache={})
    state = EnvState(tmp_path / "capture")
    for frame in range(2):
        env.frame = frame
        with state.record_step(state={}) as step:
            tools._save_observation_artifacts(primitives, state, step)

    for step in range(2):
        for prefix in ("agentview", "agentview_high", "wrist", "navview"):
            metadata = state.load(f"{prefix}_metadata.json", step=step)
            rgb = state.load(f"{prefix}.png", step=step)
            assert metadata["extrinsic_cam2world"][0][3] == step
            assert (metadata["height"], metadata["width"]) == rgb.shape[:2]
        artifacts = state.get(step).artifacts
        assert not any("_world" in name or "_depth" in name for name in artifacts)
        assert "navview_floor.png" not in artifacts
    assert len(env.metadata_calls) == 8
