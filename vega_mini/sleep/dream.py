"""
Dream module: generative replay for VegaMini sleep cycle.
"""
import torch
from ..memory.punk import PunkMemory
from ..controller.flow import FlowSolver
from ..eval.quality import QualityModel
import numpy as np
import random

def dream_cycle(punk: PunkMemory, flow_solver: FlowSolver, quality_model: QualityModel, num_dreams=500):
    """
    Generate dreams by perturbing lighthouse vectors and testing robustness.
    """
    anchors = punk.get_live_anchors(task_id=None, top_k=1000)
    if not anchors:
        return 0
    reinforced = 0
    for _ in range(num_dreams):
        a = random.choice(anchors)
        x_dream = a['vec'] + 0.1 * torch.randn_like(a['vec'])
        z = flow_solver.solve_flow(torch.randn_like(a['vec']), x_dream, y=None, anchors=anchors)[0]
        q = quality_model(z.unsqueeze(0), a['y_context'], x_dream, stv_margin=0.7)
        if q.item() > 0.7:
            punk.reinforce_nearby(z, delta_b=0.02)
            reinforced += 1
    return reinforced
