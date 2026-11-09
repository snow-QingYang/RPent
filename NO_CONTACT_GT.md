# RoboCasa contact feedback ablation

This branch starts from official RPent commit
`a744bd5f128ebef3cb8c6736cf7cd98adff681f1`.

Both `rldx_skill` and `rldx_arm` omit `grasp_contact`, `grasp_obj`, `grasped`,
and `grasp_detected`. The RLDX execution wrapper no longer queries simulator
contact for these reports. It retains execution status, gripper position,
the command/width heuristic `held_apart`, and end-effector `peak_lift`.

`RoboCasaPrimitives.task_progress()` removes contact-derived scalar fields
before state persistence. The task-specific mapping in
`robots/robocasa/primitives.py` covers the 32 Target50 composite tasks in the
installed RoboCasa version reviewed on 2026-11-04. Twenty-five tasks have
filtered fields. The other task progress remains available. This mapping
must be reviewed when adding tasks or changing the simulator version.

For example, `ScrubCuttingBoard` omits `board_contact_timer` and `sweep_range`
but keeps `gripper_obj_far`. `PrepareCoffee.contact_check` is a distance test
and remains available despite its name. Contact-dependent container and
fixture predicates are omitted in the tasks where they occur.

The simulator, native success/termination checks, geometry, prompts, memory
mechanism, VLA frame history, action conversion, and execution budgets retain
their baseline behavior. Top-level success can still depend on contact;
this ablation removes explicit contact feedback rather than all possible
inferences about contact. Existing run files are not migrated.

Offline regression coverage is in
`tests/unit_tests/robots/robocasa/test_no_contact_feedback.py`:

```bash
python -m pytest tests/unit_tests/robots/robocasa/test_no_contact_feedback.py -q
```

The local audit records task-specific field provenance under
`.local-setup/nocontact_task_progress_audit.{json,md}`. Local setup and smoke
artifacts are excluded from Git; runtime filtering does not depend on them.

## Run an episode

Follow the [RoboCasa setup guide](docs/source-en/rst_source/simulators/robocasa.rst)
and configure a [Codex planner connection](docs/source-en/rst_source/guides/configure_planner.rst).
Activate your prepared environment and install this checkout with
`pip install -e ".[robocasa]"`. Provide simulator assets, the checkpoint at
`checkpoints/rldx-1-ft-rc365`, and the matching Astra memory at
`memory/target50/robocasa/GPT_6_astra_high`. Download the required model files
before enabling offline mode. Credentials, environments, weights, memory and
run outputs are not included in this branch.

Run one episode from the checkout with the recorded model and budgets:

```bash
unset RLDX_RESET_SEED CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID RPENT_DEPTH_DEVICE
export RLDX_MAX_CHUNKS=40 RLDX_ACTION_STEPS_PER_CHUNK=8 RLDX_SETTLE_PATIENCE=999
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

rpent --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local \
  --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/no_contact_gt/composite_seen/ScrubCuttingBoard_s1
```

Repeat for the `composite_seen` and `composite_unseen` entries in
[target50.json](robots/robocasa/eval/target50.json): 16 tasks in each group,
seeds 1–5, for 160 episodes. Both groups use CLI `--split target`; choose a
separate output directory per task and seed. `--cuda-device` selects the GPU.
Read the native outcome from each `result.json`. Keep valid task failures and
budget-exhausted episodes; retry infrastructure failures without a valid result.

Custom batch scheduling, live-join helpers and machine activation scripts are
not distributed with this branch. The CLI example does not depend on them.
