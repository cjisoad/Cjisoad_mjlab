"""Read-only gradient evidence and task-balanced reductions."""
from itertools import combinations
import torch

def task_balanced_mean(values, task_ids):
    if values.shape != task_ids.shape:
        raise ValueError('Task labels must match per-frame losses')
    return torch.stack([values[task_ids == task].mean() for task in task_ids.unique()]).mean()

class GradientMonitor:
    def __init__(self):
        self.history = {}

    def measure(self, losses, named_parameters):
        parameters = [(name, p) for name, p in named_parameters if p.requires_grad]
        if not parameters:
            return {}
        gradients = {}
        for key, loss in losses.items():
            if not loss.requires_grad:
                continue
            grad = torch.autograd.grad(loss, [p for _, p in parameters],
                                       retain_graph=True, allow_unused=True)
            gradients[key] = [torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                              for (_, p), g in zip(parameters, grad)]
        groups = {'all': list(range(len(parameters))),
                  'gru': [i for i,(n,_) in enumerate(parameters) if n.startswith('rnn.')],
                  'shared': [i for i,(n,_) in enumerate(parameters) if n.startswith('mlp.') and 'output' not in n]}
        # Actual final affine layer, independent of Sequential naming.
        mlp_weights = [i for i,(n,p) in enumerate(parameters) if n.startswith('mlp.') and p.ndim == 2]
        if mlp_weights:
            last = mlp_weights[-1]
            groups['head'] = [i for i in range(last, len(parameters)) if parameters[i][0].startswith('mlp.')]
            groups['shared'] = [i for i in groups['shared'] if i not in groups['head']]
        metrics = {}
        for group, indices in groups.items():
            if not indices:
                continue
            vectors = {k: torch.cat([v[i] for i in indices]) for k,v in gradients.items()}
            norms = {k:v.norm() for k,v in vectors.items()}
            for k,n in norms.items():
                metrics[f'{group}/norm/{k}'] = n.item()
            for a,b in combinations(vectors,2):
                if norms[a] <= 1e-12 or norms[b] <= 1e-12:
                    continue
                cosine = ((vectors[a] @ vectors[b]) / (norms[a]*norms[b])).clamp(-1,1).item()
                key = f'{group}/conflict_rate/{a}__{b}'
                count, conflicts = self.history.get(key,(0,0))
                self.history[key] = (count+1,conflicts+int(cosine<0))
                metrics[key] = self.history[key][1]/self.history[key][0]
                metrics[f'{group}/cosine/{a}__{b}'] = cosine
        return metrics

    def state_dict(self):
        return dict(self.history)

    def load_state_dict(self, state):
        self.history = dict(state)
