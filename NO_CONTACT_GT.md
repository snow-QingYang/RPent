# VLA contact feedback removal with full task progress

This experiment removes only the explicit VLA grasp/contact reports from
official RPent commit `a744bd5f128ebef3cb8c6736cf7cd98adff681f1`.
The branch is `no_contact_gt`.

Both `rldx_skill` and `rldx_arm` omit `grasp_contact`, `grasp_obj`, `grasped`,
and `grasp_detected`. Their shared `RLDXSkill.run()` no longer queries
simulator contact to produce these reports. Execution status, gripper width,
the width heuristic `held_apart`, and end-effector `peak_lift` remain available.

**All native `task_progress` is retained, including contact-derived fields.**
`primitives.py` and `env_client.py` are identical to the official base.
The simulator still computes and exposes native task success. VLA success
early stopping, final scoring, prompts, geometry, memory, action conversion,
frame history, and execution budgets retain baseline behavior. No RoboCasa
simulator source is modified. This experiment does not claim to remove all
contact-related information from the Planner's context.

The earlier E04 run at commit `e33f97430972135de2208a83fc27dece7af4510c`
also filtered task progress. Its 46/160 result belongs to that different
protocol and must not be reused in this experiment. Use a fresh output root.

## Evaluation

Run the 32 composite tasks in `robots/robocasa/eval/target50.json`, seeds 1–5:
80 Seen and 80 Unseen episodes. The Planner is GPT-6-Astra / xhigh with
100 turns and 3600 seconds per episode; RLDX uses 40 chunks of 8 actions.
Use the same `GPT_6_astra_high` task/global memory corpus as the baseline.

With dependencies, checkpoint, memory and Planner credentials configured:

```bash
RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8 \
python -m rpent.cli.main --robot robocasa \
  --task-name ScrubCuttingBoard --split target --seed 1 \
  --vla-model-path ./checkpoints/rldx-1-ft-rc365 --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 3600 \
  --memory-profile local --memory-dir ./memory/target50/robocasa/GPT_6_astra_high \
  --output-dir ./runs/contact-feedback-full-progress/composite_seen/ScrubCuttingBoard_s1
```

Machine-specific queue and environment launchers live outside this branch.

## Validation

```bash
python -m pytest tests/unit_tests/robots/robocasa/test_no_contact_feedback.py -q
```

The tests exercise both VLA entry points, native success early stopping,
memory continuity, full task progress, saved state and historical state reads.
