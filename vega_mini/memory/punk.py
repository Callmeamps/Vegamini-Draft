import sqlite3
import faiss
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict, Any
import pickle

class LighthouseMemory:
    def __init__(self, db_path="vega_mini/data/lighthouses.db", 
                 index_path="vega_mini/data/lighthouse_index.faiss"):
        self.db_path = db_path
        self.index_path = index_path
        self.index = faiss.read_index(index_path)
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def drop_lighthouse(self, vec: torch.Tensor, b: float, q: float, 
                       y_context: str, task_id: str) -> int:
        """Drop a new lighthouse into memory."""
        vec_bytes = vec.cpu().numpy().astype(np.float32).tobytes()
        birth_time = time.time()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO lighthouses (vec, b, q, y_context, task_id, birth, last_reinforce)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (vec_bytes, b, q, y_context, task_id, birth_time, birth_time))
        
        lighthouse_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add to FAISS index
        vec_np = vec.cpu().numpy().astype(np.float32).reshape(1, -1)
        self.index.add(vec_np)
        faiss.write_index(self.index, self.index_path)
        
        return lighthouse_id
    
    def get_live_anchors(self, task_id: str, top_k: int = 64, min_brightness: float = 0.1):
        """Get live anchors for a task, sorted by brightness."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, vec, b, q, y_context FROM lighthouses 
            WHERE task_id = ? AND b >= ?
            ORDER BY b DESC LIMIT ?
        """, (task_id, min_brightness, top_k))
        
        anchors = []
        for row in cursor.fetchall():
            lighthouse_id, vec_bytes, b, q, y_context = row
            vec = torch.from_numpy(np.frombuffer(vec_bytes, dtype=np.float32))
            anchors.append({
                'id': lighthouse_id,
                'vec': vec,
                'b': b,
                'q': q,
                'y_context': y_context
            })
        
        conn.close()
        return anchors
    
    def reinforce_nearby(self, trajectory: torch.Tensor, delta_b: float, radius: float = 0.5):
        """Reinforce lighthouses near the trajectory."""
        traj_np = trajectory.cpu().numpy().astype(np.float32)
        
        # Find nearby lighthouses using FAISS
        if len(traj_np.shape) == 1:
            traj_np = traj_np.reshape(1, -1)
            
        distances, indices = self.index.search(traj_np, k=min(100, self.index.ntotal))
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        reinforced_count = 0
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dist_row, idx_row):
                if idx != -1 and np.sqrt(dist) < radius:
                    cursor.execute("""
                        UPDATE lighthouses 
                        SET b = b + ?, last_reinforce = ?
                        WHERE id = ?
                    """, (delta_b, time.time(), int(idx + 1)))  # FAISS is 0-indexed, DB is 1-indexed
                    reinforced_count += 1
        
        conn.commit()
        conn.close()
        
        return reinforced_count
    
    def sample_live(self, k: int, weight: str = 'b*q', task_id: Optional[str] = None):
        """Sample lighthouses weighted by brightness * quality."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if task_id:
            cursor.execute("""
                SELECT id, vec, b, q, y_context, task_id FROM lighthouses 
                WHERE task_id = ? AND b >= 0.05
            """, (task_id,))
        else:
            cursor.execute("""
                SELECT id, vec, b, q, y_context, task_id FROM lighthouses 
                WHERE b >= 0.05
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Calculate weights
        weights = []
        lighthouses = []
        
        for row in rows:
            lighthouse_id, vec_bytes, b, q, y_context, task = row
            vec = torch.from_numpy(np.frombuffer(vec_bytes, dtype=np.float32))
            
            if weight == 'b*q':
                w = b * q
            elif weight == 'b':
                w = b
            else:
                w = 1.0
                
            weights.append(w)
            lighthouses.append({
                'id': lighthouse_id,
                'vec': vec,
                'b': b,
                'q': q,
                'y_context': y_context,
                'task_id': task
            })
        
        # Sample without replacement
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        sample_size = min(k, len(lighthouses))
        indices = np.random.choice(len(lighthouses), size=sample_size, 
                                 replace=False, p=weights)
        
        return [lighthouses[i] for i in indices]
    
    def decay_all(self, lambda_factor: float = 0.95):
        """Apply decay to all lighthouse brightness."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE lighthouses SET b = b * ?", (lambda_factor,))
        deleted = cursor.execute("DELETE FROM lighthouses WHERE b < 0.05").rowcount
        
        conn.commit()
        conn.close()
        
        # Rebuild FAISS index after pruning
        if deleted > 0:
            self._rebuild_faiss_index()
        
        return deleted
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from remaining lighthouses."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT vec FROM lighthouses WHERE b >= 0.05")
        rows = cursor.fetchall()
        conn.close()
        
        # Create new index
        self.index = faiss.IndexFlatL2(1024)
        
        if rows:
            vecs = np.array([np.frombuffer(row[0], dtype=np.float32) for row in rows])
            self.index.add(vecs.astype(np.float32))
        
        faiss.write_index(self.index, self.index_path)