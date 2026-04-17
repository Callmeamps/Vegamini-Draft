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

def prune_storage_keep_portfolio(cfg):
    import sqlite3, json
    from pathlib import Path
    portfolio = json.loads(Path('portfolio.json').read_text()) if Path('portfolio.json').exists() else []
    # cell_id is from the current run config
    keep_ids = set(portfolio + [cfg.get('cell_id', 'none')])

    # Path to database - assuming it's in memory/db_iter{cfg['iteration']}.sqlite as per playbook
    # but the LighthouseMemory default is vega_mini/data/lighthouses.db.
    # I'll stick to the playbook's intended structure for DOE.
    db_path = Path(f"memory/db_iter{cfg.get('iteration', 0)}.sqlite")
    if not db_path.exists():
        # Fallback to default if it doesn't exist
        db_path = Path("vega_mini/data/lighthouses.db")
    
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Keep portfolio + top 5k by b*q
    # Note: the playbook uses a simplified SQL. I'll make it more robust.
    placeholders = ','.join(['?'] * len(keep_ids))
    cur.execute(f"""
        DELETE FROM lighthouses
        WHERE cell_id NOT IN ({placeholders})
        AND id NOT IN (SELECT id FROM lighthouses ORDER BY b*q DESC LIMIT 5000)
    """, list(keep_ids))
    
    cur.execute("DELETE FROM lighthouses WHERE b < 0.05")
    conn.commit()
    cur.execute("VACUUM")
    conn.close()

    # Checkpoints: keep only portfolio
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        for p in checkpoint_dir.glob("*.pt"):
            if not any(kid in p.name for kid in keep_ids if kid != 'none'):
                p.unlink()
