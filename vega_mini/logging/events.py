"""
Structured event and metrics logging for Vega Mini.

This module provides a unified interface for logging structured events (JSONL) 
and numeric metrics (CSV) during day and sleep cycles. It handles session 
management by creating timestamped directories for each run.

Usage Example:
    from vega_mini.logging.events import logger
    
    # Log a structured event
    logger.log_event("task_start", "runner", {"task_id": "arc_001"})
    
    # Log numeric metrics for plotting
    logger.log_metrics({"loss": 0.05, "accuracy": 0.92})
"""
import json
import time
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class VegaMiniLogger:
    """
    Structured event and metrics logger for Vega Mini.
    
    Attributes:
        base_dir (Path): Root directory for all logs.
        session_id (str): Unique timestamp for the current logging session.
        session_dir (Path): Directory for the current session's logs.
        event_file (Path): Path to the JSONL event log file.
        metrics_file (Path): Path to the CSV metrics file.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initializes the logger and creates session directories.
        
        Args:
            log_dir (str): The base directory where logs will be stored.
        """
        self.base_dir = Path(log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.event_file = self.session_dir / "events.jsonl"
        self.metrics_file = self.session_dir / "metrics.csv"
        
        self.metrics_header_written = False
        
    def log_event(self, event_type: str, component: str, data: Dict[str, Any]):
        """
        Log a structured event to a JSONL file.
        
        Args:
            event_type (str): The category of the event (e.g., 'day_step_start').
            component (str): The system component emitting the event (e.g., 'memory').
            data (Dict[str, Any]): Arbitrary key-value pairs associated with the event.
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "component": component,
            **data
        }
        
        with open(self.event_file, "a") as f:
            f.write(json.dumps(event) + "\n")
            
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log numeric metrics to a CSV file for time-series analysis.
        
        Args:
            metrics (Dict[str, Any]): Dictionary of metric names and their numeric values.
        """
        metrics_with_ts = {
            "timestamp": time.time(),
            **metrics
        }
        
        mode = "a" if self.metrics_header_written else "w"
        with open(self.metrics_file, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_with_ts.keys())
            if not self.metrics_header_written:
                writer.writeheader()
                self.metrics_header_written = True
            writer.writerow(metrics_with_ts)

    def get_session_dir(self) -> Path:
        """Returns the Path to the current session directory."""
        return self.session_dir

# Global logger instance for easy import
logger = VegaMiniLogger()
