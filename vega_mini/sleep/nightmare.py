"""
Nightmare module: adversarial probe for VegaMini sleep cycle.
"""
import torch
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.flow import FlowSolver
from vega_mini.logging.events import logger
import random

def nightmare_cycle(memory: LighthouseMemory, flow_solver: FlowSolver, velocity_net, num_nightmares=20):
    """
    Generate adversarial nightmares to test anchor robustness.
    """
    anchors = memory.get_live_anchors(task_id="default", top_k=1000)
    if not anchors:
        return 0
        
    logger.log_event("nightmare_start", "sleep", {
        "num_nightmares": num_nightmares,
        "anchor_count": len(anchors)
    })
    
    slashed = 0
    for _ in range(num_nightmares):
        a = random.choice(anchors)
        # Adversarial perturbation
        x_night = a['vec'] + 0.3 * torch.randn_like(a['vec'])
        z0 = torch.randn_like(a['vec'])
        z, _ = flow_solver.solve_flow(z0, velocity_net, x_night, y=None, anchors=anchors, t_steps=6)
        
        # Simple fail criterion: large deviation from intended context (anchor)
        if torch.norm(z - a['vec']) > 1.5:
            # Slashed by decaying by a larger factor or directly decreasing b
            # Let's use reinforce_nearby with negative delta_b if supported, 
            # or just assume decay_all handles it. 
            # For now, let's just log it as a failure of robustness.
            slashed += 1
            
    logger.log_event("nightmare_end", "sleep", {
        "slashed": slashed
    })
    
    return slashed
