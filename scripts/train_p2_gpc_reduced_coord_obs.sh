#!/usr/bin/env bash
# Usage: bash scripts/train_p2_gpc_reduced_coord_obs.sh fsq|prior|PEFT_NAME [train args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage=${1:?Specify fsq, fsq_lift_foot_smooth, prior, or a PEFT experiment name}
shift
case "$stage" in
  fsq|fsq_lift_foot_smooth) experiment="examples/experiments/mimic/${stage}_reduced_coord_obs.py" ;;
  prior|target_prior_peft|steering_prior_peft|steering_prior_peft_stopfix|sft_target_prior_peft|task_target_prior_peft|task_steering_headvel_prior_peft|task_target_prior_peft_amp|task_steering_headvel_prior_peft_amp)
    experiment="examples/experiments/gpc/${stage}_reduced_coord_obs.py" ;;
  *) echo "Unknown reduced_coord_obs stage: $stage" >&2; exit 2 ;;
esac
extra=()
case "$stage" in
  prior) extra+=(--tracker-checkpoint "${TRACKER_CHECKPOINT:?Set the reduced_coord_obs FSQ checkpoint}") ;;
  *peft*) extra+=(--prior-checkpoint "${PRIOR_CHECKPOINT:?Set the reduced_coord_obs prior checkpoint}") ;;
esac
if [[ "$stage" == sft_target_prior_peft ]]; then
  extra+=(--tracker-checkpoint "${TRACKER_CHECKPOINT:?Set the reduced_coord_obs FSQ checkpoint}")
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
unset DISPLAY
python_bin=${PYTHON_BIN:-/data/chenguanting/.venvs/protomotions-isaaclab/bin/python}
motion_file=${MOTION_FILE:-/data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_slurmrank.pt}
# The shard resolver expands _slurmrank to one existing _N.pt file per rank.
command=("$python_bin" -u protomotions/train_agent.py --robot-name p2 --simulator isaaclab
  --experiment-path "$experiment" --experiment-name "p2_${stage}_reduced_coord_obs_deploy_v3"
  --motion-file "$motion_file" --ngpu "${NGPU:-8}" --nodes 1
  --num-envs "${NUM_ENVS:-4096}" --batch-size "${BATCH_SIZE:-16384}" --headless
  "${extra[@]}" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
