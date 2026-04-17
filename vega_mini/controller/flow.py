import torch
import torch.nn as nn
import einops

class FlowSolver:
    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma
        
    def compute_lighthouse_pull(self, z: torch.Tensor, anchors: list, 
                               y_context: str = None, similarity_threshold: float = 0.5):
        """Compute the pull force from lighthouse anchors."""
        if not anchors:
            return torch.zeros_like(z)
            
        pull = torch.zeros_like(z)
        total_influence = 0.0
        
        for anchor in anchors:
            # Context similarity check (simplified for v0.1)
            if y_context is not None and anchor.get('y_context'):
                # Simple hash comparison - in practice you'd use embeddings
                if hash(y_context) != hash(anchor['y_context']):
                    continue
                    
            # Distance-based lighthouse influence
            anchor_vec = anchor['vec'].to(z.device)
            distance = torch.norm(z - anchor_vec, dim=-1, keepdim=True)
            
            # Gaussian influence with brightness weighting
            influence = anchor['b'] * torch.exp(-distance**2 / (2 * self.sigma**2))
            direction = anchor_vec - z
            
            pull += influence * direction
            total_influence += influence.item()
            
        # Normalize by total influence to prevent explosive gradients
        if total_influence > 0:
            pull = pull / max(1.0, total_influence)
            
        return pull
    
    def solve_flow(self, z0: torch.Tensor, velocity_model: nn.Module, 
                  x: torch.Tensor, y: torch.Tensor = None, 
                  anchors: list = None, t_steps: int = 6) -> tuple:
        """
        Solve flow matching with lighthouse anchoring.
        Returns (final_z, trajectory)
        """
        z = z0.clone()
        trajectory = [z.clone()]
        dt = 1.0 / t_steps
        
        for step in range(t_steps):
            t = torch.tensor(step * dt, device=z.device, dtype=z.dtype)
            
            # Get velocity from model
            v_model = velocity_model(z, t, x, y)
            
            # Add lighthouse pull
            v_lighthouse = self.compute_lighthouse_pull(z, anchors or [])
            
            # Combined velocity
            v_total = v_model + v_lighthouse
            
            # Euler step
            z = z + v_total * dt
            trajectory.append(z.clone())
            
        return z, torch.stack(trajectory, dim=0)
    
    def find_stable_points(self, trajectory: torch.Tensor, energy_threshold: float = None):
        """Find low-energy points along trajectory for lighthouse dropping."""
        # Compute energy as velocity magnitude
        velocities = torch.diff(trajectory, dim=0)
        energies = torch.norm(velocities, dim=-1)
        
        if energy_threshold is None:
            energy_threshold = torch.median(energies)
            
        stable_mask = energies < energy_threshold
        stable_points = trajectory[1:][stable_mask]  # Skip initial point
        
        return stable_points[:3]  # Return up to 3 most stable points