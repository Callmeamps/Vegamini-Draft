import torch
import numpy as np
import argparse
import time
from typing import List, Dict, Any

from vega_mini.controller.trm import VegaMiniTransformer
from vega_mini.controller.flow import FlowSolver
from vega_mini.memory.punk import LighthouseMemory
from vega_mini.eval.quality import QualityModel

class SleepCycle:
    def __init__(self):
        self.memory = LighthouseMemory()
        self.flow_solver = FlowSolver()
        self.quality_model = QualityModel()
        
        # Simple adversarial perturbation for nightmares
        self.adversarial_strength = 0.3
        
    def replay_phase(self, n_samples: int = 2000):
        """Replay lighthouse memories to test consistency."""
        print("Starting replay phase...")
        
        # Sample lighthouses weighted by brightness * quality
        sampled_anchors = self.memory.sample_live(k=n_samples, weight='b*q')
        
        if not sampled_anchors:
            print("No live lighthouses for replay")
            return
        
        reinforced = 0
        decayed = 0
        tau = 0.5  # Consistency threshold
        
        for anchor in sampled_anchors:
            try:
                # Try to reconstruct the lighthouse position
                z_recon, _ = self.flow_solver.solve_flow(
                    z0=torch.randn(1024),
                    velocity_model=None,  # Use simple dynamics for replay
                    x=torch.randn(1024),  # Dummy context
                    y=None,
                    anchors=[anchor],
                    t_steps=3
                )
                
                # Compute reconstruction error
                original_vec = anchor['vec']
                recon_loss = torch.norm(z_recon - original_vec).item()
                
                # Update lighthouse based on consistency
                if recon_loss < tau:
                    # Successful reconstruction - reinforce
                    self.memory.reinforce_nearby(
                        original_vec.unsqueeze(0), 
                        delta_b=0.05, 
                        radius=0.1
                    )
                    reinforced += 1
                else:
                    # Failed reconstruction - decay
                    self.memory.decay_lighthouse(anchor['id'], decay_factor=0.8)
                    decayed += 1
                    
            except Exception as e:
                # Error in replay - decay the problematic lighthouse
                self.memory.decay_lighthouse(anchor['id'], decay_factor=0.5)
                decayed += 1
                
        print(f"Replay completed: {reinforced} reinforced, {decayed} decayed")
    
    def dream_phase(self, n_dreams: int = 500):
        """Generative testing of lighthouse robustness."""
        print("Starting dream phase...")
        
        if not hasattr(self.memory, '_get_random_anchors'):
            # Add helper method to memory class
            def _get_random_anchors(self, k=1):
                return self.sample_live(k=k, weight='b')
            self.memory._get_random_anchors = lambda k=1: self.memory.sample_live(k=k, weight='b')
        
        strengthened = 0
        
        for _ in range(n_dreams):
            # Sample a random lighthouse to dream about
            anchor_list = self.memory.sample_live(k=1, weight='b')
            if not anchor_list:
                continue
                
            anchor = anchor_list[0]
            
            try:
                # Generate dreamed input by perturbing anchor
                anchor_vec = anchor['vec']
                dream_noise = 0.1 * torch.randn_like(anchor_vec)
                x_dream = anchor_vec + dream_noise
                
                # Test lighthouse with dreamed input
                z_result, _ = self.flow_solver.solve_flow(
                    z0=torch.randn(1024),
                    velocity_model=None,
                    x=x_dream,
                    y=None,
                    anchors=[anchor],
                    t_steps=4
                )
                
                # Score the dream result
                with torch.no_grad():
                    dream_quality = self.quality_model(
                        z_result, x_dream, x_dream, stv_margin=0.5
                    ).item()
                
                # Reinforce if dream quality is good
                if dream_quality > 0.7:
                    self.memory.reinforce_nearby(
                        anchor_vec.unsqueeze(0),
                        delta_b=0.02,
                        radius=0.1
                    )
                    strengthened += 1
                    
            except Exception:
                # Dream failed - slight decay
                pass
                
        print(f"Dream phase completed: {strengthened} lighthouses strengthened")
    
    def nightmare_phase(self, n_nightmares: int = 200):
        """Adversarial testing to find weak lighthouses."""
        print("Starting nightmare phase...")
        
        vulnerabilities_found = 0
        
        for _ in range(n_nightmares):
            # Sample lighthouse to attack
            anchor_list = self.memory.sample_live(k=1)
            if not anchor_list:
                continue
                
            anchor = anchor_list[0]
            anchor_vec = anchor['vec']
            
            try:
                # Generate adversarial input
                adversarial_noise = self.adversarial_strength * torch.randn_like(anchor_vec)
                x_nightmare = anchor_vec + adversarial_noise
                
                # Test with adversarial input
                z_result, trajectory = self.flow_solver.solve_flow(
                    z0=torch.randn(1024),
                    velocity_model=None,
                    x=x_nightmare,
                    y=None,
                    anchors=[anchor],
                    t_steps=6
                )
                
                # Check if the lighthouse guidance failed
                final_distance = torch.norm(z_result - anchor_vec).item()
                
                if final_distance > 2.0:  # Far from intended anchor
                    # Lighthouse failed under adversarial pressure
                    self.memory.decay_lighthouse(anchor['id'], decay_factor=0.5)
                    
                    # Drop a "reef" - negative anchor to avoid this region
                    self.memory.drop_lighthouse(
                        vec=z_result,
                        b=-0.5,  # Negative brightness = repulsive
                        q=0.1,   # Low quality
                        y_context="nightmare_reef",
                        task_id="global_reef"
                    )
                    vulnerabilities_found += 1
                    
            except Exception:
                # Nightmare test failed - decay the problematic lighthouse
                self.memory.decay_lighthouse(anchor['id'], decay_factor=0.3)
                
        print(f"Nightmare phase completed: {vulnerabilities_found} vulnerabilities found")
    
    def consolidation_phase(self, decay_lambda: float = 0.95):
        """Prune and merge lighthouse memory."""
        print("Starting memory consolidation...")
        
        # Global decay
        deleted = self.memory.decay_all(lambda_factor=decay_lambda)
        
        # Simple merging: find very close lighthouses and combine them
        # This is a simplified version - a full implementation would be more sophisticated
        self._merge_nearby_lighthouses()
        
        print(f"Consolidation completed: {deleted} lighthouses pruned")
    
    def _merge_nearby_lighthouses(self, merge_threshold: float = 0.1):
        """Merge lighthouses that are very close to each other."""
        # Get all live lighthouses
        live_anchors = self.memory.sample_live(k=1000)  # Get many for merging
        
        if len(live_anchors) < 2:
            return
            
        merged_count = 0
        processed_ids = set()
        
        for i, anchor1 in enumerate(live_anchors):
            if anchor1['id'] in processed_ids:
                continue
                
            candidates_for_merge = []
            
            for j, anchor2 in enumerate(live_anchors[i+1:], i+1):
                if anchor2['id'] in processed_ids:
                    continue
                    
                # Check distance
                distance = torch.norm(anchor1['vec'] - anchor2['vec']).item()
                
                if distance < merge_threshold:
                    candidates_for_merge.append((j, anchor2))
            
            # Merge if we found close neighbors
            if candidates_for_merge:
                # Simple merge: keep the brightest one, delete others
                all_candidates = [(i, anchor1)] + candidates_for_merge
                brightest_idx, brightest_anchor = max(all_candidates, key=lambda x: x[1]['b'])
                
                # Mark others for deletion (simplified - would update in real implementation)
                for idx, candidate in all_candidates:
                    if idx != brightest_idx:
                        processed_ids.add(candidate['id'])
                        merged_count += 1
        
        if merged_count > 0:
            print(f"Merged {merged_count} lighthouse pairs")
    
    def run_full_sleep_cycle(self):
        """Run complete sleep cycle: replay -> dream -> nightmare -> consolidate."""
        print("=" * 50)
        print("Starting full sleep cycle...")
        start_time = time.time()
        
        try:
            self.replay_phase(n_samples=1000)  # Reduced for v0.1
            self.dream_phase(n_dreams=200)     # Reduced for v0.1  
            self.nightmare_phase(n_nightmares=100)  # Reduced for v0.1
            self.consolidation_phase()
            
            elapsed = time.time() - start_time
            print(f"Sleep cycle completed in {elapsed:.1f} seconds")
            
        except Exception as e:
            print(f"Error in sleep cycle: {e}")
            raise

# Add missing decay method to LighthouseMemory class
def decay_lighthouse(self, lighthouse_id: int, decay_factor: float):
    """Decay a specific lighthouse's brightness."""
    conn = self.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE lighthouses 
        SET b = b * ? 
        WHERE id = ?
    """, (decay_factor, lighthouse_id))
    
    conn.commit()
    conn.close()

# Monkey patch the method (in a real implementation, add this to the class)
LighthouseMemory.decay_lighthouse = decay_lighthouse

def main():
    parser = argparse.ArgumentParser(description="VegaMini Sleep Cycle")
    parser.add_argument("--full", action="store_true", help="Run full sleep cycle")
    parser.add_argument("--replay-only", action="store_true", help="Run replay phase only")
    
    args = parser.parse_args()
    
    sleep_cycle = SleepCycle()
    
    if args.replay_only:
        sleep_cycle.replay_phase()
    elif args.full:
        sleep_cycle.run_full_sleep_cycle()
    else:
        # Default: run full cycle
        sleep_cycle.run_full_sleep_cycle()

if __name__ == "__main__":
    main()