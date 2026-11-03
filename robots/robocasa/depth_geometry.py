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

"""RGB-derived geometry using DA3 metric depth and known camera calibration."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from rpent.session import EnvState

CAMERA_GEOMETRY_IMAGES = {
    "agentview": ("agentview.png", "agentview_high.png"),
    "navview": ("navview.png", None),
    "wrist": ("wrist.png", None),
}
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "checkpoints/DA3METRIC-LARGE"
_PROCESS_RES = 504
_GEOMETRY_LOCK = threading.RLock()


def _calibration(
    intrinsic: np.ndarray, cam2world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    cam2world = np.asarray(cam2world, dtype=np.float64)
    if intrinsic.shape != (3, 3) or cam2world.shape != (4, 4):
        raise ValueError("expected 3x3 intrinsic and 4x4 camera-to-world matrices")
    if not np.isfinite(intrinsic).all() or not np.isfinite(cam2world).all():
        raise ValueError("camera calibration must be finite")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return intrinsic, cam2world


def depth_to_world(
    depth: np.ndarray, intrinsic: np.ndarray, cam2world: np.ndarray
) -> np.ndarray:
    """Back-project metric camera Z-depth; pixels are top-down [row, col]."""
    intrinsic, cam2world = _calibration(intrinsic, cam2world)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    rows, cols = np.indices(depth.shape)
    pixels = np.stack((cols, rows, np.ones_like(rows)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsic).T
    valid = np.isfinite(depth) & (depth > 0)
    camera_xyz = rays * np.where(valid, depth, np.nan)[..., None]
    return camera_xyz @ cam2world[:3, :3].T + cam2world[:3, 3]


@dataclass
class DepthGeometry:
    """One observation's predicted depth and its contemporaneous calibration."""

    depth: np.ndarray
    intrinsic: np.ndarray
    cam2world: np.ndarray

    def xyz_at(self, row: int, col: int) -> np.ndarray:
        depth = float(self.depth[row, col])
        if not np.isfinite(depth) or depth <= 0:
            return np.full(3, np.nan)
        ray = np.linalg.solve(self.intrinsic, [col, row, 1.0])
        return self.cam2world[:3, :3] @ (depth * ray) + self.cam2world[:3, 3]

    def world_map(self) -> np.ndarray:
        return depth_to_world(self.depth, self.intrinsic, self.cam2world)


def _metric_depth_from_prediction(
    raw_depth: np.ndarray, intrinsic: np.ndarray, image_shape: tuple[int, ...]
) -> np.ndarray:
    """Scale DA3METRIC canonical depth to meters and restore the RGB grid.

    DA3's single metric model requires depth * processed_focal / 300. The
    upper_bound_resize preprocessor resizes without cropping; its final grid
    can differ from the original image in both dimensions.
    """
    import cv2

    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    if raw_depth.ndim != 2 or not raw_depth.size:
        raise ValueError("Depth Anything returned an invalid depth shape")
    height, width = image_shape[:2]
    pred_h, pred_w = raw_depth.shape
    focal = (intrinsic[0, 0] * pred_w / width + intrinsic[1, 1] * pred_h / height) / 2
    valid = np.isfinite(raw_depth) & (raw_depth > 0)
    metric = np.where(valid, raw_depth, 0) * (focal / 300.0)
    if (pred_h, pred_w) != (height, width):
        metric = cv2.resize(metric, (width, height), interpolation=cv2.INTER_LINEAR)
        # Do not interpolate through missing / invalid depth predictions.
        valid = (
            cv2.resize(
                valid.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            >= 1.0 - 1e-6
        )
    return np.where(valid, metric, np.nan).astype(np.float32)


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str):
    from depth_anything_3.api import DepthAnything3

    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"DA3METRIC-LARGE checkpoint not found: {model_path}; "
            "set RPENT_DEPTH_MODEL_PATH to its local directory"
        )
    return DepthAnything3.from_pretrained(model_path).to(device).eval()


def _model_path() -> str:
    return str(
        Path(os.environ.get("RPENT_DEPTH_MODEL_PATH", str(_DEFAULT_MODEL))).resolve()
    )


def _predict_metric_depth(
    rgb: np.ndarray, intrinsic: np.ndarray, *, device: str | None = None
) -> np.ndarray:
    """Run the official metric model on RGB only; K/pose stay outside the model."""
    device = os.environ.get("RPENT_DEPTH_DEVICE") or device or "cuda:0"
    with _GEOMETRY_LOCK:
        model = _load_model(_model_path(), device)
        prediction = model.inference(
            image=[rgb],
            process_res=_PROCESS_RES,
            process_res_method="upper_bound_resize",
        )
    return _metric_depth_from_prediction(prediction.depth[0], intrinsic, rgb.shape)


def load_geometry(
    state: EnvState, *, step: int, camera: str, resolution: str
) -> DepthGeometry:
    """Load or predict depth for saved RGB, without consulting simulator depth.

    Both geometry tools share this per-observation cache. Failures propagate
    to the tool; there is deliberately no simulator geometry fallback.
    """
    if resolution not in {"low", "high"}:
        raise ValueError(f"unsupported resolution: {resolution}")
    image_name = CAMERA_GEOMETRY_IMAGES[camera][resolution == "high"]
    if image_name is None:
        raise ValueError(f"{camera} has no {resolution}-resolution image")
    prefix = Path(image_name).stem
    meta_name = f"{prefix}_metadata.json"
    depth_name = f"{prefix}_depth_predicted.npz"
    provenance_name = f"{prefix}_depth_prediction.json"
    recipe = {
        "source": "depth-anything/DA3METRIC-LARGE",
        "model_path": _model_path(),
        "process_res": _PROCESS_RES,
        "process_res_method": "upper_bound_resize",
        "depth_units": "meters",
        "scale": "processed_mean_focal / 300",
    }
    with _GEOMETRY_LOCK:
        record = state.get(step)
        for name in (image_name, meta_name):
            if name not in record.artifacts:
                raise ValueError(f"{name} not recorded for step {step}")
        metadata = state.load(meta_name, step=step)
        intrinsic, cam2world = _calibration(
            metadata["intrinsic"], metadata["extrinsic_cam2world"]
        )
        rgb = state.load(image_name, step=step)
        if (metadata["height"], metadata["width"]) != rgb.shape[:2]:
            raise ValueError("saved camera calibration does not match RGB dimensions")
        cached = all(
            name in record.artifacts and state.exists(name, step=step)
            for name in (depth_name, provenance_name)
        )
        if cached and state.load(provenance_name, step=step) == recipe:
            depth = state.load(depth_name, step=step)
        else:
            device = None
            if state.exists("depth_geometry_config.json", step=None):
                device = state.load("depth_geometry_config.json", step=None)["device"]
            depth = _predict_metric_depth(rgb, intrinsic, device=device)
            if depth.shape != rgb.shape[:2]:
                raise ValueError("predicted depth does not match RGB dimensions")
            depth = np.asarray(depth, dtype=np.float32)
            if state.save(depth_name, depth, step=step) is None:
                raise RuntimeError("failed to save predicted depth")
            if state.save(provenance_name, recipe, step=step) is None:
                raise RuntimeError("failed to save depth provenance")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("cached depth does not match RGB dimensions")
        return DepthGeometry(depth, intrinsic, cam2world)
