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

`eval_composite_astra.sh` records the local eight-GPU evaluation launcher.
Before using it on another machine, adapt the checkout and EGL paths in
`activate_env.sh`, install the RoboCasa environment, and provide the checkpoint,
assets, and `memory/target50/robocasa/GPT_6_astra_high` corpus. The launcher also
expects a local Codex profile at `.local-setup/codex-nocontactgt/`; its
`config.toml` and `auth.json` are intentionally excluded from Git. The scripts
do not install these dependencies or download the experiment inputs.
