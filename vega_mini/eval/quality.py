
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from typing import List, Tuple, Dict, Any
import random
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import config

class QualityModel(nn.Module):
    """
    Tiny 3-layer MLP for quality prediction.
    Input: [z, embed(y), embed(x), stv_margin]
    Output: scalar quality score [0,1]
    """
    
    def __init__(self, z_dim=None, hidden_dim=None):
        super().__init__()
        self.z_dim = z_dim if z_dim is not None else config.MODEL_DIM
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.QUALITY_HIDDEN_DIM
        
        # Simple text embedding for y and x contexts
        self.text_embed = nn.Embedding(config.QUALITY_VOCAB_SIZE, config.QUALITY_EMBED_DIM)
        
        # MLP layers: z(1024) + y_embed(64) + x_embed(64) + margin(1) = 1153 input
        input_dim = self.z_dim + config.QUALITY_EMBED_DIM + config.QUALITY_EMBED_DIM + 1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Training buffer for online learning
        self.buffer = []
        self.buffer_size = config.QUALITY_BUFFER_SIZE
        
    def simple_hash_embed(self, text_data, device):
        """Convert text to embedding indices via simple hash"""
        if isinstance(text_data, str):
            # Simple hash of string
            hash_val = abs(hash(text_data)) % 10000
            return torch.tensor([hash_val], device=device)
        elif isinstance(text_data, (list, tuple)):
            # Handle batch of strings
            hash_vals = [abs(hash(str(item))) % 10000 for item in text_data]
            return torch.tensor(hash_vals, device=device)
        else:
            # Fallback for other types
            hash_val = abs(hash(str(text_data))) % 10000
            return torch.tensor([hash_val], device=device)
    
    def forward(self, z, y_context, x_context, stv_margin):
        """
        Args:
            z: latent vectors [batch_size, z_dim]
            y_context: text contexts (list of strings or strings)
            x_context: input contexts (list of strings or strings)  
            stv_margin: STV margins [batch_size] or scalar
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Handle y_context embedding
        if isinstance(y_context, (list, tuple)):
            y_indices = self.simple_hash_embed(y_context, device)
        else:
            # Single string, replicate for batch
            y_indices = self.simple_hash_embed([y_context] * batch_size, device)
        
        y_embed = self.text_embed(y_indices)  # [batch_size, 64]
        
        # Handle x_context embedding  
        if isinstance(x_context, (list, tuple)):
            x_indices = self.simple_hash_embed(x_context, device)
        else:
            # Single string, replicate for batch
            x_indices = self.simple_hash_embed([x_context] * batch_size, device)
            
        x_embed = self.text_embed(x_indices)  # [batch_size, 64]
        
        # Handle margin tensor - fix the bug here
        if isinstance(stv_margin, (int, float)):
            # Single scalar, expand to batch
            margin_tensor = torch.full((batch_size, 1), stv_margin, device=device, dtype=z.dtype)
        elif torch.is_tensor(stv_margin):
            if stv_margin.dim() == 0:  # scalar tensor
                margin_tensor = stv_margin.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
            elif stv_margin.dim() == 1:  # 1D tensor
                margin_tensor = stv_margin.unsqueeze(1)  # [batch_size, 1]
            else:
                margin_tensor = stv_margin  # assume already correct shape
        else:
            # List or other iterable
            margin_tensor = torch.tensor(stv_margin, device=device, dtype=z.dtype).unsqueeze(1)
        
        # Concatenate all features
        features = torch.cat([z, y_embed, x_embed, margin_tensor], dim=1)
        
        # Forward through MLP
        quality = self.layers(features)
        return quality.squeeze(-1)  # [batch_size]
    
    def add_experience(self, z, y_context, x_context, stv_margin, quality_label):
        """Add training experience to buffer"""
        self.buffer.append({
            'z': z.detach().cpu(),
            'y': y_context,
            'x': x_context, 
            'margin': stv_margin,
            'label': quality_label
        })
        
        # Keep buffer size manageable
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def train_step(self, optimizer):
        """Train on buffer data"""
        if len(self.buffer) < 32:
            return 0.0
        
        # Sample batch from buffer
        batch_size = min(32, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        
        # Prepare batch tensors
        z_batch = torch.stack([item['z'] for item in batch])
        y_batch = [item['y'] for item in batch]
        x_batch = [item['x'] for item in batch]
        margins = torch.tensor([item['margin'] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        
        # Move to device
        device = next(self.parameters()).device
        z_batch = z_batch.to(device)
        margins = margins.to(device)
        labels = labels.to(device)
        
        # Forward pass
        self.train()
        predictions = self.forward(z_batch, y_batch, x_batch, margins)
        
        # Loss and backprop
        loss = nn.functional.mse_loss(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


def bootstrap_quality_model(model, n_samples=1000):
    """Bootstrap the quality model with synthetic data"""
    print("Bootstrapping quality model...")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device
    
    # Generate synthetic training data
    for i in range(n_samples):
        # Random latent vector
        z = torch.randn(1, 1024, device=device)
        
        # Random contexts
        y_context = f"synthetic_y_{i % 100}"
        x_context = f"synthetic_x_{i % 100}"
        
        # Random margin and quality (correlated)
        margin = random.uniform(0, 1)
        # Higher margin -> higher quality with noise
        quality = min(1.0, max(0.0, margin + random.gauss(0, 0.2)))
        
        # Add to experience buffer
        model.add_experience(z, y_context, x_context, margin, quality)
    
    # Train for a few epochs
    total_loss = 0
    n_steps = 100
    for step in range(n_steps):
        loss = model.train_step(optimizer)
        total_loss += loss
        
        if step % 20 == 0:
            print(f"Bootstrap step {step}/{n_steps}, loss: {loss:.4f}")
    
    avg_loss = total_loss / n_steps
    print(f"Bootstrap completed. Average loss: {avg_loss:.4f}")
    
    model.eval()


def generate_synthetic_sample():
    """Generate one synthetic training sample"""
    z = torch.randn(1024)  # Random latent vector
    
    # Random synthetic contexts
    task_types = ['arc', 'math', 'logic', 'visual', 'text']
    y_context = random.choice(task_types) + f"_answer_{random.randint(1,100)}"
    x_context = random.choice(task_types) + f"_input_{random.randint(1,100)}"
    
    # Random STV margin
    stv_margin = random.uniform(0.0, 1.0)
    
    # Quality correlated with margin + noise
    base_quality = stv_margin * 0.7 + 0.1  # 0.1 to 0.8 base
    noise = random.gauss(0, 0.15)
    quality = max(0.0, min(1.0, base_quality + noise))
    
    return {
        'z': z,
        'y_context': y_context,
        'x_context': x_context,
        'stv_margin': stv_margin,
        'quality': quality
    }


def create_synthetic_dataset(n_samples):
    """Create synthetic dataset for supervised training"""
    samples = []
    for _ in range(n_samples):
        samples.append(generate_synthetic_sample())
    return samples


def train_quality_model_supervised(model, training_data, n_epochs=100, lr=1e-3, batch_size=32):
    """Train quality model in supervised fashion on synthetic data"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    
    # Split into train/val
    n_train = int(0.8 * len(training_data))
    train_data = training_data[:n_train]
    val_data = training_data[n_train:]
    
    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
    
    best_val_loss = float('inf')
    best_acc = 0.0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        # Shuffle training data
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch
            z_batch = torch.stack([item['z'] for item in batch]).to(device)
            y_batch = [item['y_context'] for item in batch]
            x_batch = [item['x_context'] for item in batch]
            margins = torch.tensor([item['stv_margin'] for item in batch], dtype=torch.float32).to(device)
            labels = torch.tensor([item['quality'] for item in batch], dtype=torch.float32).to(device)
            
            # Forward pass
            predictions = model(z_batch, y_batch, x_batch, margins)
            loss = nn.functional.mse_loss(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        if epoch % 10 == 0:
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    
                    z_batch = torch.stack([item['z'] for item in batch]).to(device)
                    y_batch = [item['y_context'] for item in batch]
                    x_batch = [item['x_context'] for item in batch]
                    margins = torch.tensor([item['stv_margin'] for item in batch], dtype=torch.float32).to(device)
                    labels = torch.tensor([item['quality'] for item in batch], dtype=torch.float32).to(device)
                    
                    predictions = model(z_batch, y_batch, x_batch, margins)
                    val_loss = nn.functional.mse_loss(predictions, labels)
                    val_losses.append(val_loss.item())
                    
                    # Binary accuracy (threshold at 0.5)
                    pred_binary = (predictions > 0.5).float()
                    label_binary = (labels > 0.5).float()
                    correct += (pred_binary == label_binary).sum().item()
                    total += len(batch)
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            accuracy = correct / total if total > 0 else 0
            
            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, acc={accuracy:.3f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_acc = accuracy
    
    return best_acc