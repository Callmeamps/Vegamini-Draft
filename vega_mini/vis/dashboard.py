"""
Visualization Dashboard for Vega Mini research prototype.

This module provides tools for visualizing the internal dynamics of the 
Vega Mini system, including trajectory analysis using PCA and lighthouse 
lifecycle timelines.

Usage Example:
    from vega_mini.vis.dashboard import dashboard
    import torch
    
    # Plot candidate trajectories from a swarm
    z_trajs = [torch.randn(1024) for _ in range(32)]
    anchors = [{"vec": torch.randn(1024), "b": 1.0}]
    dashboard.plot_trajectories(z_trajs, anchors, task_id="test")
"""
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Optional
import torch

class VegaMiniDashboard:
    """
    Visualization tools for inspecting the Vega Mini system.
    
    Attributes:
        output_dir (Path): The directory where HTML plots will be saved.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initializes the dashboard and ensures the output directory exists.
        
        Args:
            output_dir (str): Base path for saving visualizations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_trajectories(self, z_trajectories: List[torch.Tensor], 
                          anchors: List[Dict[str, Any]] = None, 
                          task_id: str = "default"):
        """
        Creates a 2D PCA plot of latent z-trajectories and active memory anchors.
        
        Args:
            z_trajectories (List[torch.Tensor]): List of candidate vectors from the swarm.
            anchors (List[Dict[str, Any]], optional): List of active lighthouse anchors.
            task_id (str): Identifier for the task to include in the plot title.
        """
        if not z_trajectories:
            return
            
        z_np = torch.stack(z_trajectories).cpu().detach().numpy()
        n_samples, z_dim = z_np.shape
        
        all_vecs = z_np
        labels = ["candidate"] * n_samples
        
        if anchors:
            anchor_vecs = torch.stack([a['vec'] for a in anchors]).cpu().detach().numpy()
            all_vecs = np.concatenate([all_vecs, anchor_vecs])
            labels.extend(["anchor"] * len(anchors))
            
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(all_vecs)
        
        df = pd.DataFrame({
            "x": z_pca[:, 0],
            "y": z_pca[:, 1],
            "type": labels
        })
        
        fig = px.scatter(df, x="x", y="y", color="type", 
                         title=f"Trajectories & Anchors (PCA) - Task {task_id}")
        
        save_path = self.output_dir / f"trajectories_{task_id}_{len(z_trajectories)}.html"
        fig.write_html(str(save_path))
        
    def plot_lighthouse_timeline(self, events: List[Dict[str, Any]], task_id: str = "default"):
        """
        Creates a scatter plot showing the lifecycle (drop/reinforce/decay) 
        of lighthouses over time.
        
        Args:
            events (List[Dict[str, Any]]): List of raw event logs from JSONL.
            task_id (str): Identifier for the task.
        """
        df = pd.DataFrame(events)
        if df.empty or "event_type" not in df:
            return
            
        fig = px.scatter(df, x="timestamp", y="id", color="event_type", 
                         size="b" if "b" in df else None,
                         hover_data=["q", "y_context"] if "q" in df else None,
                         title=f"Lighthouse Timeline - Task {task_id}")
        
        save_path = self.output_dir / f"timeline_{task_id}.html"
        fig.write_html(str(save_path))
        
    def plot_metrics(self, metrics_file: str, task_id: str = "default"):
        """
        Plots metrics (e.g., quality, loss) from a CSV metrics file.
        
        Args:
            metrics_file (str): Path to the metrics CSV file.
            task_id (str): Identifier for the task.
        """
        if not Path(metrics_file).exists():
            return
            
        df = pd.read_csv(metrics_file)
        if df.empty:
            return
            
        # Plot quality over time
        if "quality" in df:
            fig = px.line(df, x="timestamp", y="quality", title=f"Quality over time - {task_id}")
            save_path = self.output_dir / f"quality_{task_id}.html"
            fig.write_html(str(save_path))
            
# Global dashboard instance
dashboard = VegaMiniDashboard()
