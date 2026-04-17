"""
Persistent memory management using SQLite and FAISS.

This module implements the 'Lighthouse' memory system, which stores high-quality 
latent vectors (lighthouses) in a SQLite database for metadata and a FAISS 
index for fast similarity searching.
"""
import sqlite3
import faiss
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict, Any
import pickle
from vega_mini.logging.events import logger

class LighthouseMemory:
    """
    Manages the lifecycle of lighthouses in persistent storage.

    Lighthouses are reinforced when they help generate high-quality answers 
    and decayed periodically to prune stale or incorrect memories.
    """
    def __init__(self, db_path="vega_mini/data/lighthouses.db", 
                 index_path="vega_mini/data/lighthouse_index.faiss"):
        """
        Initializes the memory system by loading the FAISS index.

        Args:
            db_path (str): Path to the SQLite database file.
            index_path (str): Path to the FAISS index file.
        """
        self.db_path = db_path
        self.index_path = index_path
        self.index = faiss.read_index(index_path)

    def get_connection(self):
        """Returns a new SQLite connection."""
        return sqlite3.connect(self.db_path)

    def drop_lighthouse(self, vec: torch.Tensor, b: float, q: float, 
                       y_context: str, task_id: str, cell_id: str = 'none') -> int:
        """
        Persists a new lighthouse vector into memory.

        Args:
            vec (torch.Tensor): The 1024-dim latent vector.
            b (float): Initial brightness (importance).
            q (float): Quality score at birth.
            y_context (str): Contextual identifier (e.g., hash of the answer).
            task_id (str): Identifier for the task domain.
            cell_id (str): Experiment cell identifier.

        Returns:
            int: The unique database ID of the new lighthouse.
        """
        vec_bytes = vec.detach().cpu().numpy().astype(np.float32).tobytes()
        birth_time = time.time()

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO lighthouses (vec, b, q, y_context, task_id, cell_id, birth, last_reinforce)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (vec_bytes, b, q, y_context, task_id, cell_id, birth_time, birth_time))

        lighthouse_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Add to FAISS index
        vec_np = vec.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        dim = vec_np.shape[1]
        if self.index.ntotal > 0 and self.index.d != dim:
             # Dimension mismatch, need to rebuild or reset
             self.index = faiss.IndexFlatL2(dim)
        elif self.index.ntotal == 0 and self.index.d != dim:
             self.index = faiss.IndexFlatL2(dim)
             
        self.index.add(vec_np)
        faiss.write_index(self.index, self.index_path)

        logger.log_event("lighthouse_drop", "memory", {
            "id": lighthouse_id,
            "b": b,
            "q": q,
            "task_id": task_id,
            "y_context": y_context[:16] if y_context else None
        })

        return lighthouse_id

    def get_live_anchors(self, task_id: str, top_k: int = 64, min_brightness: float = 0.1):
        """
        Retrieves active lighthouses for a given task.

        Args:
            task_id (str): The task to retrieve anchors for.
            top_k (int): Maximum number of anchors to return.
            min_brightness (float): Minimum brightness threshold.

        Returns:
            List[Dict]: A list of anchor dictionaries containing 'vec', 'b', etc.
        """
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

        logger.log_event("get_anchors", "memory", {
            "task_id": task_id,
            "count": len(anchors),
            "min_brightness": min_brightness
        })

        return anchors

    def reinforce_nearby(self, trajectory: torch.Tensor, delta_b: float, radius: float = 0.5):
        """
        Increases brightness of lighthouses near a trajectory.

        Args:
            trajectory (torch.Tensor): Vector(s) representing the successful path.
            delta_b (float): Amount to increase brightness.
            radius (float): Distance threshold for reinforcement.
        """
        traj_np = trajectory.detach().cpu().numpy().astype(np.float32)

        # Find nearby lighthouses using FAISS
        if len(traj_np.shape) == 1:
            traj_np = traj_np.reshape(1, -1)

        if self.index.ntotal == 0:
            return 0

        distances, indices = self.index.search(traj_np, k=min(100, self.index.ntotal))

        conn = self.get_connection()
        cursor = conn.cursor()

        reinforced_ids = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dist_row, idx_row):
                if idx != -1 and np.sqrt(dist) < radius:
                    lighthouse_id = int(idx + 1) # FAISS is 0-indexed, DB is 1-indexed
                    cursor.execute("""
                        UPDATE lighthouses 
                        SET b = b + ?, last_reinforce = ?
                        WHERE id = ?
                    """, (delta_b, time.time(), lighthouse_id))
                    reinforced_ids.append(lighthouse_id)

        conn.commit()
        conn.close()

        if reinforced_ids:
            logger.log_event("reinforce", "memory", {
                "count": len(reinforced_ids),
                "ids": reinforced_ids,
                "delta_b": delta_b,
                "radius": radius
            })

        return len(reinforced_ids)

    def sample_live(self, k: int, weight: str = 'b*q', task_id: Optional[str] = None):
        """
        Samples lighthouses for sleep cycle replay.

        Args:
            k (int): Number of lighthouses to sample.
            weight (str): SQL expression for sampling weights.
            task_id (str, optional): Filter by task.
        """
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

        sampled = [lighthouses[i] for i in indices]

        logger.log_event("sample_live", "memory", {
            "count": len(sampled),
            "ids": [s['id'] for s in sampled],
            "task_id": task_id
        })

        return sampled

    def decay_all(self, lambda_factor: float = 0.95):
        """
        Applies decay to all lighthouses and prunes those with low brightness.

        Args:
            lambda_factor (float): Decay multiplier (0.0 to 1.0).

        Returns:
            int: Number of lighthouses pruned.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("UPDATE lighthouses SET b = b * ?", (lambda_factor,))

        # Get IDs that will be deleted
        cursor.execute("SELECT id FROM lighthouses WHERE b < 0.05")
        deleted_ids = [row[0] for row in cursor.fetchall()]

        deleted = cursor.execute("DELETE FROM lighthouses WHERE b < 0.05").rowcount

        conn.commit()
        conn.close()

        # Rebuild FAISS index after pruning
        if deleted > 0:
            self._rebuild_faiss_index()

        logger.log_event("decay", "memory", {
            "lambda_factor": lambda_factor,
            "deleted_count": deleted,
            "deleted_ids": deleted_ids
        })

        return deleted

    def _rebuild_faiss_index(self):
        """Rebuilds the FAISS index from scratch using active lighthouses."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT vec FROM lighthouses WHERE b >= 0.05")
        rows = cursor.fetchall()
        conn.close()

        if rows:
            vecs = np.array([np.frombuffer(row[0], dtype=np.float32) for row in rows])
            dim = vecs.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vecs.astype(np.float32))
        else:
            # Fallback to default dim or keep existing if empty
            pass

        faiss.write_index(self.index, self.index_path)