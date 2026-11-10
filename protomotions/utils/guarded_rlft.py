"""Stage-three valid-only PPO/AMP updates with matching DDP collectives."""
import logging
import torch
from protomotions.utils.valid_samples import valid_moments, current_valid_mask

log = logging.getLogger(__name__)


def enabled(agent):
    return getattr(agent.env.simulator, '_physics_guard', None) is not None


def install_guard(agent):
    if not getattr(agent.config, 'physics_guard_enabled', True):
        log.info('PHYSICS_GUARD disabled by configuration')
        return
    if agent.env.simulator.__class__.__module__ == 'protomotions.simulator.mjlab.simulator':
        from protomotions.simulator.mjlab.physics_guard import MjLabPhysicsGuard
        agent.env.simulator._physics_guard = MjLabPhysicsGuard(
            agent.env.simulator, env=agent.env, folder=agent.root_dir / 'physics_guard',
            joint_speed_limit=getattr(agent.config, 'physics_guard_joint_speed_limit', 1000.),
            max_consecutive=getattr(agent.config, 'physics_guard_max_consecutive', 3))
        log.info('PHYSICS_GUARD enabled for mjlab')
        return
    if agent.env.simulator.__class__.__module__ != 'protomotions.simulator.isaaclab.simulator':
        return
    from protomotions.utils.physics_guard import PhysicsGuard
    agent.env.simulator._physics_guard = PhysicsGuard(
        agent.env.simulator, env=agent.env, folder=agent.root_dir / 'physics_guard',
        joint_speed_limit=getattr(agent.config, 'physics_guard_joint_speed_limit', 1000.),
        max_consecutive=getattr(agent.config, 'physics_guard_max_consecutive', 3))
    log.info('PHYSICS_GUARD enabled: speed_limit=%s rad/s, max_consecutive=%s, logs=%s',
             agent.env.simulator._physics_guard.limit,
             agent.env.simulator._physics_guard.max_consecutive,
             agent.env.simulator._physics_guard.folder)


def moments(agent, values, valid):
    x = values[valid].double()
    packed = torch.stack((x.new_tensor(x.numel()), x.sum(), x.square().sum()))
    packed = agent.fabric.all_reduce(packed, reduce_op='sum')
    n, total, square = packed
    mean = total / n.clamp_min(1)
    var = (square / n.clamp_min(1) - mean.square()).clamp_min(0)
    return mean.to(values.dtype), var.sqrt().to(values.dtype), n


def clear_invalid(agent, mask, step=None):
    buffer = agent.experience_buffer
    for key in buffer.store_dict:
        if key in ('physics_valid', 'dones'):
            continue
        data = getattr(buffer, key)
        if step is not None:
            data = data[step]
        data[~mask] = 0


def record_guarded(agent, callback, next_obs, actions, rewards, dones, terminated,
                   done_indices, extras, step):
    if not enabled(agent) or getattr(agent, '_guard_recording', False):
        return callback(next_obs, actions, rewards, dones, terminated, done_indices, extras, step)
    valid = extras['raw/physics_valid']
    agent.experience_buffer.update_data('physics_valid', step, valid)
    agent._guard_recording = True
    try:
        with valid_moments(valid):
            result = callback(next_obs, actions, rewards, dones, terminated,
                              done_indices[valid[done_indices]], extras, step)
        clear_invalid(agent, valid, step)
        norms = [getattr(agent, 'running_reward_norm', None)]
        amp = getattr(agent, 'amp_component', None)
        if amp is not None:
            norms.append(amp.running_reward_norm)
            amp.num_cumulative_bad_transitions[~valid] = 0
        for norm in norms:
            if norm is not None and norm.discounted_reward is not None:
                norm.discounted_reward[~valid] = 0
        return result
    finally:
        agent._guard_recording = False


def optimization_guarded(agent, callback, batch, idx):
    if not enabled(agent) or getattr(agent, '_guard_optimizing', False):
        return callback(batch, idx)
    # Reset even when the first minibatch contains no valid samples anywhere.
    if idx == 0:
        agent._kl_early_stop_triggered = False
    mask = batch['physics_valid'].bool()
    count = mask.sum().float()
    global_count = agent.fabric.all_reduce(count.clone(), reduce_op='sum')
    if global_count.item() == 0:
        return {'physics_guard/skipped_empty_batch': count.new_tensor(1.),
                'physics_guard/valid_samples': count}
    data = {k: v[mask] for k, v in batch.items() if k != 'physics_valid'}
    if count.item() == 0:
        data = {k: torch.zeros_like(v[:1]) for k, v in batch.items() if k != 'physics_valid'}
    forward_valid = torch.full((len(data['action']),), bool(count.item()), device=mask.device)
    agent._guard_optimizing = True
    agent._guard_optimizer_scales = {}
    agent._guard_loss_scale = agent.fabric.world_size * count / global_count
    try:
        with valid_moments(forward_valid):
            result = callback(data, idx)
        result['physics_guard/valid_samples'] = count
        result['physics_guard/skipped_empty_batch'] = count.new_tensor(0.)
        return result
    finally:
        agent._guard_optimizing = False
        agent._guard_optimizer_scales = {}


class PhysicsGuardRLFTMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_guard(self)

    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        if enabled(self):
            self.experience_buffer.register_key('physics_valid', dtype=torch.bool)

    def record_rollout_step(self, *args, **kwargs):
        return record_guarded(self, super().record_rollout_step, *args, **kwargs)

    def pre_process_dataset(self):
        if not enabled(self):
            return super().pre_process_dataset()
        # A rejected transition has zero TD residual and done=1. This cuts the
        # GAE recursion without leaking its reward/state into earlier valid steps.
        valid = self.experience_buffer.physics_valid
        clear_invalid(self, valid)
        computed = self.compute_advantages()  # Includes AMP's separate value head.
        for key, value in computed.items():
            value[~valid] = 0
            self.experience_buffer.batch_update_data(key, value)
        adv = self.experience_buffer.advantages
        mean, std, count = moments(self, adv, valid)
        logs = {'adv_norm/raw_mean': mean, 'adv_norm/raw_std': std}
        cfg = self.config.advantage_normalization
        if cfg.enabled and count.item() > 0:
            if cfg.use_ema:
                safe_std = self.adv_std_ema.clamp_min(cfg.min_std)
                normalized = (adv - self.adv_mean_ema) / safe_std if cfg.shift_mean else adv / safe_std
                clipped = normalized.clamp(-cfg.clamp_range, cfg.clamp_range)
                denorm = clipped * safe_std + (self.adv_mean_ema if cfg.shift_mean else 0)
                new_mean, new_std, _ = moments(self, denorm, valid)
                self.adv_mean_ema = cfg.ema_alpha * new_mean + (1-cfg.ema_alpha) * self.adv_mean_ema
                self.adv_std_ema = cfg.ema_alpha * (new_std + 1e-8) + (1-cfg.ema_alpha) * self.adv_std_ema
                adv = clipped
            else:
                adv = (adv - mean) / (std + 1e-8) if cfg.shift_mean else adv / (std + 1e-8)
        adv[~valid] = 0
        self.experience_buffer.batch_update_data('advantages', adv)
        self._adv_norm_log = logs

    def perform_optimization_step(self, batch_dict, batch_idx):
        return optimization_guarded(self, super().perform_optimization_step, batch_dict, batch_idx)

    def _step_optimizer(self, loss, model, optimizer, model_name):
        if getattr(self, '_guard_optimizing', False):
            loss = loss * self._guard_optimizer_scales.get(model_name, self._guard_loss_scale)
        return super()._step_optimizer(loss=loss, model=model, optimizer=optimizer, model_name=model_name)

    def discriminator_step(self, batch_dict):
        if not getattr(self, '_guard_optimizing', False):
            return super().discriminator_step(batch_dict)
        size = min(len(batch_dict['action']), self.config.amp_parameters.discriminator_batch_size)
        valid = current_valid_mask()[:size]
        count = valid.sum().float()
        total = self.fabric.all_reduce(count.clone(), reduce_op='sum')
        self._guard_optimizer_scales['discriminator'] = self.fabric.world_size * count / total.clamp_min(1)
        with valid_moments(valid):
            # AMPAgentMixin precedes this class in the MRO; call the component
            # directly so discriminator-sized normalization uses the correct mask.
            return self.amp_component.discriminator_step(batch_dict)


class PhysicsGuardAMPOuterMixin:
    """Wrap AMP's post-PPO work in the same valid-sample transaction."""
    def record_rollout_step(self, *args, **kwargs):
        return record_guarded(self, super().record_rollout_step, *args, **kwargs)

    def perform_optimization_step(self, batch_dict, batch_idx):
        return optimization_guarded(self, super().perform_optimization_step, batch_dict, batch_idx)

    def discriminator_step(self, batch_dict):
        if not getattr(self, "_guard_optimizing", False):
            return super().discriminator_step(batch_dict)
        return PhysicsGuardRLFTMixin.discriminator_step(self, batch_dict)
