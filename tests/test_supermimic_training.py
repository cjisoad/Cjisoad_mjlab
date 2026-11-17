"""Training plumbing and terminal-safe multistep replay."""
import torch

def test_replay_masks_terminal_boundaries_and_retains_terminal_transition():
    from src.tasks.legmanip_distil.supermimic.training import TransitionReplay
    replay = TransitionReplay(capacity=8)
    for t in range(4):
        replay.add({'state':torch.full((2,37),float(t)),
                    'next_state':torch.full((2,37),float(t+1)),
                    'targets':torch.zeros(2,4,12),
                    'contacts':torch.zeros(2,4),
                    'collision':torch.zeros(2,1),
                    'done':torch.tensor([t==1,False]),
                    'owner':torch.tensor([2,0])})
    sample = replay.sample(32, 4)
    mask = sample['valid']
    assert mask[0].all()
    assert not mask[2:, sample['env_ids']==0].any()
    assert (sample['next_state'][1,:,0]==2).all()

def test_registry_exposes_fresh_supermimic_task():
    import src.tasks
    from mjlab.tasks.registry import load_rl_cfg
    cfg = load_rl_cfg('legmanip_distil_supermimic')
    assert cfg.experiment_name == 'legmanip_distil_supermimic'
    assert not cfg.resume
    assert cfg.supermimic.horizon == 4


def test_replay_excludes_nonfinite_terminal_data_without_nan_gradients():
    from src.tasks.legmanip_distil.supermimic.training import TransitionReplay
    replay = TransitionReplay(capacity=4)
    replay.add({'state':torch.zeros(1,37), 'next_state':torch.tensor([[float('nan'),float('inf'),-float('inf')] + [0.]*34]),
                'targets':torch.zeros(1,4,12), 'contacts':torch.zeros(1,4),
                'collision':torch.zeros(1,1), 'done':torch.tensor([True]),
                'owner':torch.tensor([0])})
    sample = replay.sample(2,1)
    assert not sample['valid'].any()
    assert torch.isfinite(sample['next_state']).all()
    assert not sample['next_state'].any()

def test_masked_invalid_rotation_cannot_poison_multistep_model_gradients():
    from src.tasks.legmanip_distil.supermimic.physics import WorldModel
    model = WorldModel(hidden_dim=16)
    state = torch.zeros(1,37)
    state[:,3] = 1.
    targets = torch.zeros(1,4,12)
    predicted = model(state,targets)['state']
    invalid_terminal = torch.zeros_like(state)
    loss = model.loss(predicted,targets,invalid_terminal,torch.zeros(1,4),
                      torch.zeros(1,1),valid_mask=torch.zeros(1,dtype=torch.bool))['total']
    loss.backward()
    assert loss == 0.
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
