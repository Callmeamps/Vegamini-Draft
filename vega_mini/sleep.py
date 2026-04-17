"""
Sleep cycle orchestrator for VegaMini.
Runs replay, dream, nightmare, prune, and merge steps.
"""
from memory.punk import PunkMemory
from controller.flow import FlowSolver
from eval.quality import QualityModel
from sleep.dream import dream_cycle
from sleep.nightmare import nightmare_cycle
from sleep.consolidate import consolidate_cycle


def night_cycle():
    punk = PunkMemory()
    flow_solver = FlowSolver()
    quality_model = QualityModel()

    print("[Sleep] Starting replay/consolidate...")
    survived = consolidate_cycle(punk, flow_solver)
    print(f"[Sleep] Survived anchors: {survived}")

    print("[Sleep] Starting dreams...")
    reinforced = dream_cycle(punk, flow_solver, quality_model)
    print(f"[Sleep] Reinforced anchors (dreams): {reinforced}")

    print("[Sleep] Starting nightmares...")
    slashed = nightmare_cycle(punk, flow_solver)
    print(f"[Sleep] Slashed anchors (nightmares): {slashed}")

    print("[Sleep] Sleep cycle complete.")

if __name__ == "__main__":
    night_cycle()
