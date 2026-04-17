"""
Consolidate module: replay, prune, and merge for VegaMini sleep cycle.
"""
import torch
from ..memory.punk import PunkMemory
from ..controller.flow import FlowSolver
import numpy as np
import time

def consolidate_cycle(punk: PunkMemory, flow_solver: FlowSolver, tau=0.5, decay_lambda=0.95):
    """
    Replay, prune, and merge lighthouses.
    """
    anchors = punk.get_live_anchors(task_id=None, top_k=2000)
    survived = 0
    for a in anchors:
        z = flow_solver.solve_flow(a['vec'], x=None, y=a['y_context'], anchors=[])[0]
        loss = torch.norm(z - a['vec'])
        if loss < tau:
            punk.reinforce(a['id'], 0.05)
            survived += 1
        else:
            punk.decay(a['id'], 0.2)
    punk.decay_all(lambda_=decay_lambda)
    punk.delete_where('b < 0.05')
    punk.merge_nearby(delta=0.1)
    return survived
