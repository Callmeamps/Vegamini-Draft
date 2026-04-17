import torch
import sqlite3
from dataclasses import dataclass
from typing import Optional, List, Any
import time

@dataclass
class Lighthouse:
    """A single lighthouse point in latent space"""
    id: int
    vec: torch.Tensor  # 1024-dim vector
    b: float  # brightness
    q: float  # quality at birth
    y_context: str  # hash of y
    task_id: str
    birth: float  # timestamp
    last_reinforce: float
    
    @classmethod
    def from_db_row(cls, row, vec_tensor):
        """Create Lighthouse from SQLite row + tensor"""
        return cls(
            id=row[0],
            vec=vec_tensor,
            b=row[2],
            q=row[3],
            y_context=row[4],
            task_id=row[5],
            birth=row[6],
            last_reinforce=row[7]
        )

class LighthouseManager:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lighthouses (
                id INTEGER PRIMARY KEY,
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
    
    def add_lighthouse(self, vec: torch.Tensor, b: float, q: float, 
                      y_context: str, task_id: str) -> int:
        """Add new lighthouse to database"""
        now = time.time()
        vec_bytes = vec.cpu().numpy().astype('float16').tobytes()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO lighthouses (vec, b, q, y_context, task_id, birth, last_reinforce)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (vec_bytes, b, q, y_context, task_id, now, now))
        
        lighthouse_id = cursor.lastrowid
        self.conn.commit()
        return lighthouse_id
    
    def get_lighthouses(self, task_id: str, min_brightness: float = 0.05) -> List[Lighthouse]:
        """Get all lighthouses for a task above brightness threshold"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, vec, b, q, y_context, task_id, birth, last_reinforce
            FROM lighthouses 
            WHERE task_id = ? AND b >= ?
            ORDER BY b DESC
        """, (task_id, min_brightness))
        
        lighthouses = []
        for row in cursor.fetchall():
            vec_bytes = row[1]
            vec = torch.from_numpy(
                np.frombuffer(vec_bytes, dtype='float16').astype('float32')
            )