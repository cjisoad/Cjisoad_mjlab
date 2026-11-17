"""Real-data value calibration and independent physical model validation."""
import math
import torch
from torch.nn import functional as F
from .physics import state_delta

@torch.no_grad()
def validate_dynamics(algorithm):
    a, c = algorithm, algorithm.sm_cfg
    horizon = min(c.max_horizon, len(a.replay.steps))
    batch = a.replay.sample(c.validation_batch_size, horizon, allowed_envs=a.validation_envs)
    predicted = batch['state'][0]
    errors, contact_errors, velocity_errors = [], [], []
    for t in range(horizon):
        output = a.world_model(predicted, batch['targets'][t])
        predicted = output['state']
        expected = torch.where(batch['valid'][t,:,None], batch['next_state'][t], predicted)
        errors.append((state_delta(predicted, expected) / a.world_model.delta_scale).square().mean(-1).sqrt())
        contact_errors.append((output['contact_logits'].sigmoid()-batch['contacts'][t]).abs().mean(-1))
        velocity_errors.append((predicted[:,7:10]-expected[:,7:10]).square().mean(-1).sqrt())
    error, contact, velocity = map(torch.stack,(errors,contact_errors,velocity_errors))
    task = batch.get('task', batch['owner'])
    stats = {}
    checked = torch.zeros_like(a.validation_counts,dtype=torch.bool)
    for k in task.unique().tolist():
        mask = (task==k) & batch['valid']
        if not mask.any():
            continue
        score = error[mask].mean().item()
        endpoint = (task[-1] == k) & batch['valid'][-1]
        checked[k] = endpoint.any() and horizon == c.max_horizon
        endpoint_score = error[-1,endpoint].mean().item() if endpoint.any() else float('inf')
        contact_score = contact[-1,endpoint].mean().item() if endpoint.any() else 1.
        previous = a.validation_scores[k].item()
        a.validation_scores[k] = .8*previous+.2*endpoint_score if torch.isfinite(a.validation_scores[k]) and endpoint.any() else endpoint_score
        a.contact_scores[k] = contact_score
        good = (endpoint.any() and endpoint_score < c.validation_error_scale
                and a.validation_scores[k] < c.validation_error_scale and contact_score < .25
                and horizon == c.max_horizon)
        a.validation_counts[k] = a.validation_counts[k]+1 if good else 0
        stats[f'supermimic/model_validation/task_{k}/normalized_rmse'] = score
        stats[f'supermimic/model_validation/task_{k}/endpoint_samples'] = endpoint.sum().item()
        if endpoint.any():
            stats[f'supermimic/model_validation/task_{k}/endpoint_rmse'] = endpoint_score
        stats[f'supermimic/model_validation/task_{k}/velocity_rmse'] = velocity[mask].mean().item()
        stats[f'supermimic/model_validation/task_{k}/contact_mae'] = contact[mask].mean().item()
    # Missing/all-invalid tasks interrupt consecutive evidence and revoke an
    # earlier promotion. Never let an old good task remain trusted forever.
    a.validation_counts[~checked] = 0
    a.validation_scores[~checked] = float('inf')
    a.contact_scores[~checked] = 1.
    for k in range(len(checked)):
        if not checked[k]:
            stats[f'supermimic/model_validation/task_{k}/endpoint_samples'] = 0.
    # Extend only after every represented behavior has repeated independent
    # successful validations. A missing task never silently counts as reliable.
    reliable = (a.validation_counts >= c.reliable_validations) & (a.validation_scores < c.validation_error_scale) & (a.contact_scores < .25)
    a.active_horizon = c.max_horizon if reliable.all() else c.horizon
    stats['supermimic/active_horizon'] = float(a.active_horizon)
    return stats

def train_real_critic(algorithm):
    a, c = algorithm, algorithm.sm_cfg
    batch = a.replay.sample(c.model_batch_size, c.model_horizon, allowed_envs=a.training_envs)
    x = batch['real_features']
    with torch.no_grad():
        next_values = a.critic_target(batch['next_features']).squeeze(-1)
        returns = real_lambda_returns(batch['real_reward'],next_values,batch['done'],
                                      batch['valid'],c.gamma,c.gae_lambda)
    predictions = a.critic(x).squeeze(-1)
    valid = batch['valid']
    loss = (F.smooth_l1_loss(predictions, returns, reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    a.critic_optimizer.zero_grad(set_to_none=True)
    (c.real_critic_weight*loss).backward()
    torch.nn.utils.clip_grad_norm_(a.critic.parameters(),1.)
    a.critic_optimizer.step()
    with torch.no_grad():
        for target, p in zip(a.critic_target.parameters(),a.critic.parameters()):
            target.lerp_(p,.01)
    return {'supermimic/critic_loss':loss.item(),
            'supermimic/real_value_error':((predictions.detach()-returns).abs()*valid).sum().item()/valid.sum().clamp_min(1).item()}

def real_lambda_returns(reward, next_values, done, valid, gamma, lam):
    """An invalid future row cuts recursion, preserving immediate bootstrap."""
    carry = next_values[-1]
    returns = []
    for t in reversed(range(len(reward))):
        continuation = carry
        if t+1 < len(reward):
            continuation = torch.where(valid[t+1],carry,next_values[t])
        bootstrap = (1-lam)*next_values[t]+lam*continuation
        carry = reward[t]+gamma*(~done[t].bool()).float()*bootstrap
        returns.append(carry)
    return torch.stack(returns[::-1])


def imagined_termination(env, scenario_done_before=None):
    intent = env.command_manager.get_term('intent')
    maximum = getattr(env,'max_episode_length',math.ceil(env.cfg.episode_length_s/env.step_dt))
    scenario = intent.finished | intent.timed_out if scenario_done_before is None else scenario_done_before
    return scenario | (env.episode_length_buf >= maximum)
