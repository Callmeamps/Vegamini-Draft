
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class QualityModel(nn.Module):
    """
    Quality model Q_φ: predicts quality score given [z, embed(y), embed(x), stv_margin]
    Architecture: 3-layer MLP
    Input: [z, y_embed, x_embed, stv_margin]
    Output: scalar quality score
    """
    def __init__(self, z_dim=None, y_embed_dim=None, x_embed_dim=None):
        super().__init__()
        self.z_dim = z_dim if z_dim is not None else config.MODEL_DIM
        self.y_embed_dim = y_embed_dim if y_embed_dim is not None else config.QUALITY_EMBED_DIM
        self.x_embed_dim = x_embed_dim if x_embed_dim is not None else config.QUALITY_EMBED_DIM
        # Simple text embedding (hash-based for bootstrap)
        self.y_embedder = nn.Embedding(config.QUALITY_VOCAB_SIZE, self.y_embed_dim)
        self.x_embedder = nn.Embedding(config.QUALITY_VOCAB_SIZE, self.x_embed_dim)
        input_dim = self.z_dim + self.y_embed_dim + self.x_embed_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Quality score between 0-1
        )
    def hash_text(self, text):
        """Simple hash function for text to embedding indices"""
        if isinstance(text, str):
            return hash(text) % config.QUALITY_VOCAB_SIZE
        elif isinstance(text, (list, tuple)):
            return [hash(str(item)) % config.QUALITY_VOCAB_SIZE for item in text]
        else:
            return hash(str(text)) % config.QUALITY_VOCAB_SIZE

    def forward(self, z, y, x, stv_margin):
        """
        Args:
            z: [batch, z_dim] - latent representation
            y: [batch] or list - output/answer text
            x: [batch] or list - input text
            stv_margin: [batch] or [batch, 1] - STV voting margin
        Returns:
            quality: [batch] - predicted quality score
        """
        batch_size = z.shape[0] if len(z.shape) > 1 else 1
        device = z.device
        # Embed y
        if isinstance(y, (list, tuple)):
            y_hash = torch.tensor([self.hash_text(item) for item in y], dtype=torch.long, device=device)
        else:
            y_hash = torch.tensor([self.hash_text(y)] * batch_size, dtype=torch.long, device=device)
        # Embed x
        if isinstance(x, (list, tuple)):
            x_hash = torch.tensor([self.hash_text(item) for item in x], dtype=torch.long, device=device)
        else:
            x_hash = torch.tensor([self.hash_text(x)] * batch_size, dtype=torch.long, device=device)
        y_embed = self.y_embedder(y_hash)
        x_embed = self.x_embedder(x_hash)
        # Margin
        if isinstance(stv_margin, (float, int)):
            stv_margin = torch.full((batch_size, 1), float(stv_margin), device=device)
        elif torch.is_tensor(stv_margin):
            if stv_margin.dim() == 0:
                stv_margin = stv_margin.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
            elif stv_margin.dim() == 1:
                stv_margin = stv_margin.unsqueeze(1)
        else:
            stv_margin = torch.tensor(stv_margin, device=device, dtype=z.dtype).unsqueeze(1)
        # Concatenate all features
        features = torch.cat([z, y_embed, x_embed, stv_margin], dim=-1)
        # Predict quality
        quality = self.mlp(features)
        return quality.squeeze(-1)


def generate_synthetic_data(n_samples=1000, z_dim=1024):
    """
    Generate synthetic training data for bootstrapping the quality model
    
    Strategy:
    - High quality: z close to some "good" anchors, coherent y, reasonable margin
    - Low quality: random z, incoherent y, low margins
    """
    
    # Create some "good anchor" templates
    good_anchors = torch.randn(10, z_dim) * 0.5  # Less noisy good points
    
    data = []
    
    for i in range(n_samples):
        if np.random.random() < 0.6:  # 60% high quality samples
            # High quality sample
            anchor_idx = np.random.randint(0, len(good_anchors))
            z = good_anchors[anchor_idx] + torch.randn(z_dim) * 0.2  # Close to good anchor
            
            # Coherent text patterns
            y_patterns = [
                "The answer is 42",
                "Pattern: A B C D",
                "Solution: rotate clockwise",
                "Result: blue square",
                "Output: [1, 2, 3]"
            ]
            x_patterns = [
                "What is the meaning of life?",
                "Complete the sequence: A B C",
                "How to solve this puzzle?",
                "What color should this be?", 
                "Generate the next numbers"
            ]
            
            y = np.random.choice(y_patterns)
            x = np.random.choice(x_patterns)
            stv_margin = torch.tensor(0.7 + np.random.random() * 0.3)  # High margin
            quality = torch.tensor(0.8 + np.random.random() * 0.2)    # High quality
            
        else:  # 40% low quality samples
            # Low quality sample
            z = torch.randn(z_dim)  # Random point
            
            # Incoherent text
            y_bad = [
                "Error: undefined",
                "???",
                "Random gibberish xyz",
                "",
                "Failed to compute"
            ]
            x_bad = [
                "Malformed input ###",
                "Corrupt data",
                "Invalid question format",
                "???",
                ""
            ]
            
            y = np.random.choice(y_bad)
            x = np.random.choice(x_bad)
            stv_margin = torch.tensor(np.random.random() * 0.3)  # Low margin
            quality = torch.tensor(np.random.random() * 0.4)     # Low quality
        
        data.append({
            'z': z,
            'y': y,
            'x': x,
            'stv_margin': stv_margin,
            'quality': quality
        })
    
    return data


def create_dataloader(data, batch_size=32, shuffle=True):
    """Convert synthetic data to DataLoader"""
    
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def collate_fn(batch):
        z = torch.stack([item['z'] for item in batch])
        y = [item['y'] for item in batch]
        x = [item['x'] for item in batch]
        stv_margin = torch.stack([item['stv_margin'] for item in batch])
        quality = torch.stack([item['quality'] for item in batch])
        
        return z, y, x, stv_margin, quality
    
    dataset = SyntheticDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_quality_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """Train the quality model"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for z, y, x, stv_margin, quality in train_loader:
            z, stv_margin, quality = z.to(device), stv_margin.to(device), quality.to(device)
            
            optimizer.zero_grad()
            pred_quality = model(z, y, x, stv_margin)
            loss = criterion(pred_quality, quality)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(z)
            train_samples += len(z)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for z, y, x, stv_margin, quality in val_loader:
                z, stv_margin, quality = z.to(device), stv_margin.to(device), quality.to(device)
                
                pred_quality = model(z, y, x, stv_margin)
                loss = criterion(pred_quality, quality)
                
                val_loss += loss.item() * len(z)
                val_samples += len(z)
        
        train_loss /= train_samples
        val_loss /= val_samples
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/quality_model_best.pth')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Quality Model for VegaMini')
    parser.add_argument('--bootstrap', type=int, default=1000, help='Number of synthetic samples for bootstrap training')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("VegaMini Quality Model Training")
    print("=" * 50)
    print(f"Bootstrap samples: {args.bootstrap}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    data = generate_synthetic_data(n_samples=args.bootstrap, z_dim=1024)
    
    # Split into train/val
    val_size = int(len(data) * args.val_split)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing Quality Model...")
    model = QualityModel(z_dim=1024)
    
    # Train model
    print("\nStarting training...")
    trained_model = train_quality_model(
        model, train_loader, val_loader, 
        epochs=args.epochs, lr=args.lr
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), 'models/quality_model_final.pth')
    print("\nModel saved to models/quality_model_final.pth")
    
    # Test the model with some examples
    print("\nTesting model with sample predictions...")
    trained_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # High quality example
        z_good = torch.randn(1, 1024).to(device) * 0.3  # Low noise
        q_good = trained_model(z_good, ["The answer is 42"], ["What is the meaning?"], torch.tensor([0.9]).to(device))
        
        # Low quality example  
        z_bad = torch.randn(1, 1024).to(device) * 2.0   # High noise
        q_bad = trained_model(z_bad, ["???"], ["Corrupt input"], torch.tensor([0.1]).to(device))
        
        print(f"High quality example prediction: {q_good.item():.3f}")
        print(f"Low quality example prediction: {q_bad.item():.3f}")


if __name__ == '__main__':
    main()