"""
Dream module: generative replay for VegaMini sleep cycle.
"""
import torch
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.flow import FlowSolver
from vega_mini.eval.quality import QualityModel
from vega_mini.logging.events import logger
import numpy as np
import random

def dream_cycle(memory: LighthouseMemory, flow_solver: FlowSolver, velocity_net, quality_model: QualityModel, num_dreams=50):
    """
    Generate dreams by perturbing lighthouse vectors and testing robustness.
    """
    anchors = memory.get_live_anchors(task_id="default", top_k=1000)
    if not anchors:
        return 0
        
    logger.log_event("dream_start", "sleep", {
        "num_dreams": num_dreams,
        "anchor_count": len(anchors)
    })
    
    reinforced = 0
    for _ in range(num_dreams):
        a = random.choice(anchors)
        # Perturb the anchor to create a "dream" prompt
        x_dream = a['vec'] + 0.1 * torch.randn_like(a['vec'])
        # Generate answer for this dream prompt
        z0 = torch.randn_like(a['vec'])
        z, _ = flow_solver.solve_flow(z0, velocity_net, x_dream, y=None, anchors=anchors, t_steps=6)
        
        # Assess quality (simplified for dream)
        q = quality_model(z.unsqueeze(0), a['y_context'], "dream", torch.tensor([[0.7]]))
        
        if q.item() > 0.7:
            memory.reinforce_nearby(z, delta_b=0.02, radius=0.1)
            reinforced += 1
            
    logger.log_event("dream_end", "sleep", {
        "reinforced": reinforced
    })
    
    return reinforced
