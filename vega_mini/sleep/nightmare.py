"""
Nightmare module: adversarial probe for VegaMini sleep cycle.
"""
import torch
from ..memory.punk import PunkMemory
from ..controller.flow import FlowSolver
import random

def nightmare_cycle(punk: PunkMemory, flow_solver: FlowSolver, num_nightmares=200):
    """
    Generate adversarial nightmares to test anchor robustness.
    """
    anchors = punk.get_live_anchors(task_id=None, top_k=1000)
    if not anchors:
        return 0
    slashed = 0
    for _ in range(num_nightmares):
        a = random.choice(anchors)
        x_night = a['vec'] + 0.2 * torch.randn_like(a['vec'])  # Adversarial perturbation
        z = flow_solver.solve_flow(torch.randn_like(a['vec']), x_night, y=None, anchors=anchors)[0]
        # Simple fail criterion: large deviation from anchor
        if torch.norm(z - a['vec']) > 1.0:
            punk.slash(a['id'], 0.5)
            slashed += 1
    return slashed
