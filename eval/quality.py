import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityModel(nn.Module):
    """
    Tiny 3-layer MLP for quality scoring.
    Input: [z, embed(y), embed(x), stv_margin]
    Output: scalar quality score [0,1]
    """
    
    def __init__(self, z_dim=1024, text_embed_dim=512, total_dim=None):
        super().__init__()
        
        self.z_dim = z_dim
        self.text_embed_dim = text_embed_dim
        
        # Total input: z + y_embed + x_embed + stv_margin
        if total_dim is None:
            total_dim = z_dim + text_embed_dim + text_embed_dim + 1
        
        self.text_encoder = nn.Linear(100, text_embed_dim)  # simple text embedding
        
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output [0,1]
        )
        
    def encode_text(self, text):
        """Simple text encoding - just hash to vector for now"""
        if isinstance(text, str):
            # Convert string to simple numeric encoding
            text_hash = hash(text) % (2**16)
            text_vec = torch.zeros(100)
            text_vec[text_hash % 100] = 1.0
            return text_vec.unsqueeze(0)
        return text
        
    def forward(self, z, y_text, x_text, stv_margin):
        """
        Args:
            z: latent vector [batch, z_dim]
            y_text: output text (string or tensor)
            x_text: input text (string or tensor)  
            stv_margin: STV voting margin [batch, 1]
        """
        batch_size = z.shape[0]
        
        # Encode text inputs
        if isinstance(y_text, str):
            y_embed = self.text_encoder(self.encode_text(y_text).to(z.device))
            y_embed = y_embed.expand(batch_size, -1)
        else:
            y_embed = self.text_encoder(y_text)
            
        if isinstance(x_text, str):
            x_embed = self.text_encoder(self.encode_text(x_text).to(z.device))
            x_embed = x_embed.expand(batch_size, -1)
        else:
            x_embed = self.text_encoder(x_text)
        
        # Concatenate all features
        features = torch.cat([z, y_embed, x_embed, stv_margin], dim=1)
        
        # Predict quality score
        quality = self.mlp(features)
        return quality.squeeze(-1)
    
    def train_step(self, z, y_text, x_text, stv_margin, target_quality, optimizer):
        """Training step with BCE loss"""
        pred_quality = self.forward(z, y_text, x_text, stv_margin)
        loss = F.binary_cross_entropy(pred_quality, target_quality)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

def train_quality_model(model, data_loader, epochs=10):
    """Train the quality model on collected data"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in data_loader:
            z, y_text, x_text, stv_margin, target = batch
            loss = model.train_step(z, y_text, x_text, stv_margin, target, optimizer)
            total_loss += loss
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def bootstrap_quality_model(model, num_samples=1000):
    """Bootstrap with synthetic data"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i in range(num_samples):
        # Generate synthetic training data
        z = torch.randn(1, 1024)
        y_text = f"answer_{i % 10}"
        x_text = f"question_{i % 10}"
        stv_margin = torch.rand(1, 1)
        
        # Simple heuristic for quality: higher margin = higher quality
        target_quality = torch.sigmoid(stv_margin * 3).squeeze()
        
        loss = model.train_step(z, y_text, x_text, stv_margin, target_quality, optimizer)
        
        if i % 100 == 0:
            print(f"Bootstrap step {i}/{num_samples}, Loss: {loss:.4f}")