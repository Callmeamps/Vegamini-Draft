import torch
import torch.nn as nn
import einops
from .flow import FlowSolver

class VegaMiniTransformer(nn.Module):
    def __init__(self, dim: int = 1024, n_heads: int = 8, n_layers: int = 6, 
                 max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.flow_solver = FlowSolver()
        
        # Input embeddings
        self.x_embed = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads) for _ in range(n_layers)
        ])
        
        # Flow velocity network
        self.velocity_net = VelocityNetwork(dim)
        
        # Output projection
        self.y_proj = nn.Linear(dim, dim)
        
        # Layer norm
        self.ln_f = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, z0: torch.Tensor = None, 
               anchors: list = None) -> tuple:
        """
        Forward pass combining transformer reasoning with flow dynamics.
        Returns (y_output, z_final, trajectory)
        """
        batch_size, seq_len, _ = x.shape
        
        if z0 is None:
            z0 = torch.randn(batch_size, self.dim, device=x.device)
            
        # Embed inputs
        x_emb = self.x_embed(x)
        
        # Add positional encoding
        if seq_len <= self.pos_embed.size(0):
            x_emb = x_emb + self.pos_embed[:seq_len]
        
        # Transformer processing
        h = x_emb
        for layer in self.layers:
            h = layer(h)
            
        h = self.ln_f(h)
        
        # Extract context for y generation
        context = h.mean(dim=1)  # Simple pooling
        
        # Flow solve with current state
        z_final, trajectory = self.flow_solver.solve_flow(
            z0=z0,
            velocity_model=self.velocity_net,
            x=context,
            y=None,
            anchors=anchors
        )
        
        # Generate output y from final z
        y_output = self.y_proj(z_final)
        
        return y_output, z_final, trajectory

class TransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        
        return x

class VelocityNetwork(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim * 2),  # z + t + x + y -> hidden
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)  # -> velocity
        )
        
    def forward(self, z: torch.Tensor, t: torch.Tensor, 
               x: torch.Tensor, y: torch.Tensor = None):
        """Predict velocity at point z, time t, given context x, y."""
        batch_size = z.shape[0] if len(z.shape) > 1 else 1
        
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        if y is None:
            y = torch.zeros_like(x)
        elif len(y.shape) == 1:
            y = y.unsqueeze(0)
            
        # Expand time to match batch
        if t.numel() == 1:
            t = t.expand(batch_size, 1)
        elif len(t.shape) == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
            
        # Concatenate inputs
        inputs = torch.cat([z, t, x, y], dim=-1)
        
        velocity = self.net(inputs)
        
        return velocity.squeeze(0) if batch_size == 1 else velocity