"""
Sleep cycle orchestrator for VegaMini.
Runs replay, dream, nightmare, prune, and merge steps.
"""
import argparse
import json
from pathlib import Path
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.flow import FlowSolver
from vega_mini.controller.trm import VegaMiniTransformer
from vega_mini.eval.quality import QualityModel
from vega_mini.sleep.dream import dream_cycle
from vega_mini.sleep.nightmare import nightmare_cycle
from vega_mini.sleep.consolidate import consolidate_cycle, prune_storage_keep_portfolio
from vega_mini.logging.events import logger


def night_cycle(config=None, prune=False):
    cfg = {
        'model_dim': 1024,
        'decay_lambda': 0.95,
        'arch': 'trm_7m'
    }
    if config:
        cfg.update(config)
        
    # Architecture mapping
    arch_dims = {
        'trm_7m': 1024,
        'trm_14m': 2048
    }
    model_dim = arch_dims.get(cfg['arch'], cfg['model_dim'])

    memory = LighthouseMemory()
    flow_solver = FlowSolver()
    controller = VegaMiniTransformer(dim=model_dim)
    quality_model = QualityModel(z_dim=model_dim)

    logger.log_event("sleep_cycle_start", "sleep", {"cfg": cfg})

    print("[Sleep] Starting replay/consolidate...")
    survived = consolidate_cycle(memory, flow_solver, controller.velocity_net, decay_lambda=cfg.get('decay_lambda', 0.95))
    print(f"[Sleep] Survived anchors: {survived}")

    print("[Sleep] Starting dreams...")
    reinforced = dream_cycle(memory, flow_solver, controller.velocity_net, quality_model)
    print(f"[Sleep] Reinforced anchors (dreams): {reinforced}")

    print("[Sleep] Starting nightmares...")
    slashed = nightmare_cycle(memory, flow_solver, controller.velocity_net)
    print(f"[Sleep] Slashed anchors (nightmares): {slashed}")

    if prune:
        print("[Sleep] Pruning storage...")
        prune_storage_keep_portfolio(cfg)

    logger.log_event("sleep_cycle_end", "sleep", {
        "survived": survived,
        "reinforced": reinforced,
        "slashed": slashed
    })
    
    print("[Sleep] Sleep cycle complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--prune", action="store_true", help="Enable portfolio-aware pruning")
    args = parser.parse_args()
    
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            
    night_cycle(config=config, prune=args.prune)
