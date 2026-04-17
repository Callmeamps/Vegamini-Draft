"""
Sleep cycle orchestrator for VegaMini.
Runs replay, dream, nightmare, prune, and merge steps.
"""
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.flow import FlowSolver
from vega_mini.controller.trm import VegaMiniTransformer
from vega_mini.eval.quality import QualityModel
from vega_mini.sleep.dream import dream_cycle
from vega_mini.sleep.nightmare import nightmare_cycle
from vega_mini.sleep.consolidate import consolidate_cycle
from vega_mini.logging.events import logger


def night_cycle():
    memory = LighthouseMemory()
    flow_solver = FlowSolver()
    controller = VegaMiniTransformer(dim=1024) # Assume 1024 for now
    quality_model = QualityModel(z_dim=1024)

    logger.log_event("sleep_cycle_start", "sleep", {})

    print("[Sleep] Starting replay/consolidate...")
    survived = consolidate_cycle(memory, flow_solver, controller.velocity_net)
    print(f"[Sleep] Survived anchors: {survived}")

    print("[Sleep] Starting dreams...")
    reinforced = dream_cycle(memory, flow_solver, controller.velocity_net, quality_model)
    print(f"[Sleep] Reinforced anchors (dreams): {reinforced}")

    print("[Sleep] Starting nightmares...")
    slashed = nightmare_cycle(memory, flow_solver, controller.velocity_net)
    print(f"[Sleep] Slashed anchors (nightmares): {slashed}")

    logger.log_event("sleep_cycle_end", "sleep", {
        "survived": survived,
        "reinforced": reinforced,
        "slashed": slashed
    })
    
    print("[Sleep] Sleep cycle complete.")

if __name__ == "__main__":
    night_cycle()
