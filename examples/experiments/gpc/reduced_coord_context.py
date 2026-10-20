"""Compatibility checks for the P2 sensor-only checkpoint chain."""

from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
from protomotions.envs.obs.p2_deploy_sensors import current_deploy_obs


def require_reduced_checkpoint(checkpoint, stage):
    resolved = load_resolved_configs_from_checkpoint(checkpoint)
    component = resolved['env'].observation_components.get('reduced_coord_obs')
    if (component is None or component.compute_func is not current_deploy_obs
            or component.static_params.get("contract_version") != 3
            or component.static_params.get("quaternion_sign") != "preserve"
            or not component.static_params.get('include_projected_gravity', True)):
        raise ValueError(f'{stage} checkpoint must use the 71D P2 reduced_coord_obs: {checkpoint}')
    if len(component.static_params.get('motor_dof_ids', [])) != 29:
        raise ValueError(f'{stage} checkpoint does not have 29 P2 motor channels')
    model = resolved['agent'].model
    keys = (model.actor.mu_model.decoder.in_keys if stage == 'tracker'
            else model.prior.context_encoder.in_keys)
    if 'reduced_coord_obs' not in keys or 'max_coords_obs' in keys:
        raise ValueError(f'{stage} model must consume reduced_coord_obs: {checkpoint}')
    if stage == 'tracker' and model.actor.num_out != 29:
        raise ValueError('Deployment-aligned tracker must have 29 actions')
    if resolved['env']._target_ not in (
            'protomotions.envs.base_env.p2_deploy.P2DeployEnv',
            'protomotions.envs.base_env.p2_deploy.P2DeployLift2mmEnv'):
        raise ValueError('Checkpoint must use the deployment sensor environment')
    from protomotions.envs.obs.p2_deploy_sensors import CONTRACT
    robot = resolved['robot']
    for i, name in enumerate(CONTRACT['joint_names']):
        control = robot.control.control_info[name]
        if control.stiffness != CONTRACT['kp'][i] or control.damping != CONTRACT['kd'][i]:
            raise ValueError(f'Checkpoint PD parameters differ from deployment: {name}')
    return resolved
