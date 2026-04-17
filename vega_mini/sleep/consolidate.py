"""
Consolidate module: replay, prune, and merge for VegaMini sleep cycle.
"""
import torch
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.flow import FlowSolver
from vega_mini.logging.events import logger
import numpy as np
import time

def consolidate_cycle(memory: LighthouseMemory, flow_solver: FlowSolver, velocity_net, tau=0.5, decay_lambda=0.95):
    """
    Replay, prune, and merge lighthouses.
    """
    anchors = memory.get_live_anchors(task_id="default", top_k=2000)
    survived = 0
    
    logger.log_event("consolidate_start", "sleep", {
        "anchor_count": len(anchors),
        "tau": tau,
        "decay_lambda": decay_lambda
    })
    
    for a in anchors:
        # Replay: solve flow with anchors=None to test self-consistency
        # Use a['vec'] as x since we don't store original x (simple heuristic for now)
        z_final, _ = flow_solver.solve_flow(a['vec'], velocity_net, x=a['vec'], y=a['y_context'], anchors=[], t_steps=6)
        loss = torch.norm(z_final - a['vec'])
        
        if loss < tau:
            memory.reinforce_nearby(a['vec'], delta_b=0.05, radius=0.01)
            survived += 1
        
    deleted = memory.decay_all(lambda_factor=decay_lambda)
    
    logger.log_event("consolidate_end", "sleep", {
        "survived": survived,
        "deleted": deleted
    })
    
    return survived
