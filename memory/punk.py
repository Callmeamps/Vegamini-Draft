import sqlite3
import numpy as np
import torch
import faiss
from typing import List, Tuple, Optional, Dict
import pickle
import time

class Lighthouse:
    def __init__(self, id: int, vec: torch.Tensor, b: float, q: float, 
                 y_context: str, task_id: str, birth: float, last_reinforce: float):
        self.id = id
        self.vec = vec
        self.b = b
        self.q = q
        self.y_context = y_context
        self.task_id = task_id
        self.birth = birth
        self.last_reinforce = last_reinforce

class PunkMemory:
    def __init__(self, db_path: str = "lighthouses.db", index_path: str = "lighthouse.index"):
        self.db_path = db_path
        self.index_path = index_path
        self.dim = 1024
        
        # Initialize SQLite
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
        # Initialize FAISS
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = {}  # faiss_id -> lighthouse_id
        self._load_or_create_index()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lighthouses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vec BLOB,
                b REAL,
                q REAL,
                y_context TEXT,
                task_id TEXT,
                birth REAL,
                last_reinforce REAL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_task ON lighthouses(task_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_brightness ON lighthouses(b)")
        self.conn.commit()
    
    def _load_or_create_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
            # Rebuild id_map from database
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM lighthouses ORDER BY id")
            for i, (lighthouse_id,) in enumerate(cursor.fetchall()):
                self.id_map[i] = lighthouse_id
        except:
            # Create new index
            self.index = faiss.IndexFlatL2(self.dim)
            self.id_map = {}
    
    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
    
    def drop_lighthouse(self, vec: torch.Tensor, b: float, q: float, 
                       y_context: str, task_id: str) -> int:
        """Add a new lighthouse to memory"""
        vec_np = vec.detach().cpu().numpy().astype(np.float32)
        vec_blob = pickle.dumps(vec_np)
        
        cursor = self.conn.cursor()
        now = time.time()
        cursor.execute("""
            INSERT INTO lighthouses (vec, b, q, y_context, task_id, birth, last_reinforce)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (vec_blob, b, q, y_context, task_id, now, now))
        
        lighthouse_id = cursor.lastrowid
        self.conn.commit()
        
        # Add to FAISS
        self.index.add(vec_np.reshape(1, -1))
        faiss_id = self.index.ntotal - 1
        self.id_map[faiss_id] = lighthouse_id
        
        self._save_index()
        return lighthouse_id
    
    def get_live_anchors(self, task_id: Optional[str] = None, top_k: int = 64) -> List[Lighthouse]:
        """Get top lighthouses by brightness"""
        cursor = self.conn.cursor()
        
        if task_id:
            cursor.execute("""
                SELECT id, vec, b, q, y_context, task_id, birth, last_reinforce
                FROM lighthouses 
                WHERE task_id = ? AND b > 0.05
                ORDER BY b DESC 
                LIMIT ?
            """, (task_id, top_k))
        else:
            cursor.execute("""
                SELECT id, vec, b, q, y_context, task_id, birth, last_reinforce
                FROM lighthouses 
                WHERE b > 0.05
                ORDER BY b DESC 
                LIMIT ?
            """, (top_k,))
        
        lighthouses = []
        for row in cursor.fetchall():
            vec_np = pickle.loads(row[1])
            vec_tensor = torch.from_numpy(vec_np)
            lighthouse = Lighthouse(
                id=row[0], vec=vec_tensor, b=row[2], q=row[3],
                y_context=row[4], task_id=row[5], birth=row[6], last_reinforce=row[7]
            )
            lighthouses.append(lighthouse)
        
        return lighthouses
    
    def reinforce_nearby(self, trajectory: torch.Tensor, delta_b: float, threshold: float = 0.1):
        """Reinforce lighthouses near the trajectory"""
        traj_np = trajectory.detach().cpu().numpy().astype(np.float32)
        
        if self.index.ntotal == 0:
            return
        
        # Search for nearby lighthouses
        distances, indices = self.index.search(traj_np.reshape(1, -1), min(100, self.index.ntotal))
        
        for dist, faiss_id in zip(distances[0], indices[0]):
            if dist < threshold and faiss_id in self.id_map:
                lighthouse_id = self.id_map[faiss_id]
                self.reinforce(lighthouse_id, delta_b)
    
    def reinforce(self, lighthouse_id: int, delta_b: float):
        """Increase brightness of a lighthouse"""
        cursor = self.conn.cursor()
        now = time.time()
        cursor.execute("""
            UPDATE lighthouses 
            SET b = b + ?, last_reinforce = ?
            WHERE id = ?
        """, (delta_b, now, lighthouse_id))
        self.conn.commit()
    
    def decay(self, lighthouse_id: int, decay_factor: float):
        """Decrease brightness of a lighthouse"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE lighthouses 
            SET b = b * ?
            WHERE id = ?
        """, (decay_factor, lighthouse_id))
        self.conn.commit()
    
    def decay_all(self, lambda_factor: float = 0.95):
        """Decay all lighthouse brightness"""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE lighthouses SET b = b * ?", (lambda_factor,))
        self.conn.commit()
    
    def delete_where(self, condition: str):
        """Delete lighthouses matching condition"""
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM lighthouses WHERE {condition}")
        self.conn.commit()
        
        # Rebuild FAISS index
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild FAISS index from database"""
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = {}
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, vec FROM lighthouses ORDER BY id")
        
        vecs = []
        for i, (lighthouse_id, vec_blob) in enumerate(cursor.fetchall()):
            vec_np = pickle.loads(vec_blob)
            vecs.append(vec_np)
            self.id_map[i] = lighthouse_id
        
        if vecs:
            vecs_array = np.array(vecs).astype(np.float32)
            self.index.add(vecs_array)
        
        self._save_index()
    
    def sample_live(self, k: int = 1, weight: str = 'b*q') -> List[Lighthouse]:
        """Sample lighthouses weighted by brightness*quality"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT id, vec, b, q, y_context, task_id, birth, last_reinforce,
                   ({weight}) as weight
            FROM lighthouses 
            WHERE b > 0.05
            ORDER BY RANDOM()
        """)
        
        rows = cursor.fetchall()
        if not rows:
            return []
        
        # Weighted sampling
        weights = np.array([row[8] for row in rows])
        weights = weights / weights.sum()
        
        indices = np.random.choice(len(rows), size=min(k, len(rows)), 
                                 replace=False, p=weights)
        
        lighthouses = []
        for idx in indices:
            row = rows[idx]
            vec_np = pickle.loads(row[1])
            vec_tensor = torch.from_numpy(vec_np)
            lighthouse = Lighthouse(
                id=row[0], vec=vec_tensor, b=row[2], q=row[3],
                y_context=row[4], task_id=row[5], birth=row[6], last_reinforce=row[7]
            )
            lighthouses.append(lighthouse)
        
        return lighthouses
    
    def close(self):
        self.conn.close()