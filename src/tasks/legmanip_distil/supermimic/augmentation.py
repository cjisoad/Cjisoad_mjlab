"""Physical mirroring and history-aware four-leg policy symmetry."""
import torch

JOINT_SWAP = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
FOOT_SWAP = (1, 0, 3, 2)


def mirror_joints(value):
    signs = value.new_tensor((-1., 1., 1.) * 4)
    return value[..., JOINT_SWAP] * signs


def mirror_state(state):
    signs = state.new_tensor((1., -1., 1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.))
    return torch.cat((state[..., :13] * signs,
                      mirror_joints(state[..., 13:25]), mirror_joints(state[..., 25:37])), -1)


def mirror_action(action, default):
    return (mirror_joints(default + .25 * action) - default) / .25


def mirror_observation(obs, default):
    signs = obs.new_tensor((-1., 1., -1., 1., -1., 1., 1., -1., -1., 1., -1., 1.))
    return torch.cat((obs[..., :12] * signs,
                      mirror_joints(obs[..., 12:24] + default) - default,
                      mirror_joints(obs[..., 24:36]),
                      mirror_action(obs[..., 36:48], default)), -1)


def mirror_model_batch(batch):
    """Mirror half the batch, preserving the four actual substep targets."""
    mask = torch.rand(len(batch['env_ids']), device=batch['state'].device) < .5
    for key in ('state', 'next_state'):
        batch[key][:, mask] = mirror_state(batch[key][:, mask])
    batch['targets'][:, mask] = mirror_joints(batch['targets'][:, mask])
    batch['contacts'][:, mask] = batch['contacts'][:, mask][..., FOOT_SWAP]
    return batch


def sequence_symmetry(student, obs, dones, eligible, default, hidden_state=None, mirrored_hidden_state=None, *, return_errors=False):
    """Recompute both GRU histories; never transform an opaque hidden vector."""
    prediction, _ = student.forward_sequence(obs, dones, hidden_state)
    reflected, _ = student.forward_sequence(mirror_observation(obs, default), dones, mirrored_hidden_state)
    # Exclude mirror fixed points to permit the policy to initiate a gait.
    radius = (obs[..., 12:36] - mirror_observation(obs, default)[..., 12:36]).square().mean(-1).sqrt()
    gate = eligible * ((radius - .03) / .12).clamp(0., 1.)
    error = (reflected - mirror_action(prediction, default)).square().mean(-1)
    return (error, gate) if return_errors else (error * gate).sum() / gate.sum().clamp_min(1.)
