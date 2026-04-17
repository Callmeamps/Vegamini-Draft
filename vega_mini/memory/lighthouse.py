"""Lighthouse management - drop, reinforce, decay logic."""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import hashlib

@dataclass
class Lighthouse:
    id: int
    vec: torch.Tensor
    b: float  # brightness
    q: float  # quality
    y_context: str  # hash of y
    task_id: str
    birth: float
    last_reinforce: float
    
    def to_dict(self):
        return {
            'id': self.id,
            'vec': self.vec.cpu().numpy().astype(np.float16).tobytes(),
            'b': self.b,
            'q': self.q,
            'y_context': self.y_context,
            'task_id': self.task_id,
            'birth': self.birth,
            'last_reinforce': self.last_reinforce
        }
    
    @classmethod
    def from_dict(cls, data):
        vec = torch.from_numpy(np.frombuffer(data['vec'], dtype=np.float16)).float()
        return cls(
            id=data['id'],
            vec=vec,
            b=data['b'],
            q=data['q'],
            y_context=data['y_context'],
            task_id=data['task_id'],
            birth=data['birth'],
            last_reinforce=data['last_reinforce']
        )

def hash_context(y: torch.Tensor) -> str:
    """Create deterministic hash of y context."""
    if y is None:
        return "none"
    y_bytes = y.cpu().numpy().astype(np.float32).tobytes()
    return hashlib.md5(y_bytes).hexdigest()[:16]

def similarity(y1_hash: str, y2_hash: str) -> float:
    """Simple string similarity for context matching."""
    if y1_hash == "none" or y2_hash == "none":
        return 0.5  # neutral
    return 1.0 if y1_hash == y2_hash else 0.0