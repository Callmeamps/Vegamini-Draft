import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path


from vega_mini.memory.punk import LighthouseMemory
from vega_mini.controller.trm import VegaMiniTransformer
from vega_mini.controller.flow import FlowSolver
from vega_mini.eval.quality import QualityModel
from vega_mini.logging.events import logger
from vega_mini.vis.dashboard import dashboard


class VegaMiniRunner:
    def __init__(self, model_dim=1024):
        # Initialize components
        self.model_dim = model_dim
        self.memory = LighthouseMemory()
        self.controller = VegaMiniTransformer(dim=model_dim)
        self.flow_solver = FlowSolver()
        self.quality_model = QualityModel(z_dim=model_dim)
        # Training state
        self.step_count = 0
        self.quality_data = []  # Store training data for quality model
        
    def embed_text(self, text, dim=1024, device=None):
        """Simple hash-based embedding for text input."""
        if device is None:
            device = torch.device('cpu')
        # Use hash to seed a random vector for reproducibility
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        vec = rng.randn(dim).astype(np.float32)
        return torch.tensor(vec, device=device)

    def day_step(self, x_text, task_id="default"):
        """
        Single day step: swarm + STV + write
        """
        # If input is a dict (e.g., {'input': '...'}), extract the string
        if isinstance(x_text, dict) and 'input' in x_text:
            x_text = x_text['input']
        if not isinstance(x_text, str):
            x_text = str(x_text)
            
        logger.log_event("day_step_start", "runner", {
            "step": self.step_count,
            "x_text": x_text[:50],
            "task_id": task_id
        })

        # Embed x_text
        x_embed = self.embed_text(x_text, dim=self.model_dim)

        # Get live anchors
        anchors = self.memory.get_live_anchors(task_id, top_k=64)

        # Swarm: generate candidates
        y_candidates = []
        z_trajectories = []
        swarm_size = min(32, max(4, len(anchors) // 2))

        for worker_id in range(swarm_size):
            z0 = torch.randn(self.model_dim)
            z_final, _ = self.flow_solver.solve_flow(
                z0, self.controller.velocity_net, x_embed, y=None, anchors=anchors, t_steps=6
            )
            y_answer = self.controller.y_proj(z_final.unsqueeze(0)).squeeze(0)
            y_candidates.append(y_answer)
            z_trajectories.append(z_final)
            
        # STV voting
        clusters = self.cluster_answers(y_candidates)
        ballots = self.rank_by_worker(clusters, swarm_size)
        y_winner, stv_margin = self.stv_vote(ballots, y_candidates)
        
        # Find winning trajectory
        try:
            win_idx = next((i for i, y in enumerate(y_candidates) if torch.allclose(y, y_winner)), 0)
        except:
            win_idx = 0
        z_winner = z_trajectories[win_idx]
        
        # Quality assessment
        quality_score = self.assess_quality(z_winner, y_winner, x_text, stv_margin)
        
        logger.log_metrics({
            "quality": quality_score,
            "stv_margin": stv_margin,
            "swarm_size": swarm_size,
            "anchor_count": len(anchors)
        })
        
        # Write lighthouses if quality is high
        if quality_score > 0.8:
            self.write_lighthouses(z_winner, y_winner, x_text, task_id, quality_score)
        
        # Reinforce nearby anchors
        if quality_score > 0.65:
            self.memory.reinforce_nearby(z_winner, delta_b=0.1 * quality_score)
            
        # Visualization (periodically or if requested)
        if self.step_count % 10 == 0:
            dashboard.plot_trajectories(z_trajectories, anchors, task_id)
            
        self.step_count += 1
        return y_winner, quality_score
        
    def cluster_answers(self, y_candidates, similarity_threshold=0.9):
        if not y_candidates:
            return []
        y_mat = torch.stack([y if isinstance(y, torch.Tensor) else torch.tensor(y) for y in y_candidates])
        y_mat = y_mat.float()
        y_norm = y_mat / (y_mat.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.mm(y_norm, y_norm.t())
        clusters = []
        assigned = set()
        for i in range(len(y_candidates)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i+1, len(y_candidates)):
                if j not in assigned and sim_matrix[i, j] > similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)
        return clusters
        
    def rank_by_worker(self, clusters, num_workers):
        ballots = []
        for worker_id in range(num_workers):
            ranked = sorted(clusters, key=len, reverse=True)
            ballots.append(ranked)
        return ballots
        
    def stv_vote(self, ballots, y_candidates=None):
        if not ballots or not ballots[0]:
            return torch.zeros(self.model_dim), 0.0
        all_clusters = [cluster for ballot in ballots for cluster in ballot]
        if not all_clusters:
            return torch.zeros(self.model_dim), 0.0
        from collections import Counter
        cluster_tuples = [tuple(sorted(cluster)) for cluster in all_clusters]
        most_common = Counter(cluster_tuples).most_common(1)
        if not most_common:
            return torch.zeros(self.model_dim), 0.0
        winner_indices = list(most_common[0][0])
        margin = len(winner_indices) / max(sum(len(cluster) for cluster in all_clusters), 1)
        if y_candidates and winner_indices:
            return y_candidates[winner_indices[0]], margin
        return torch.zeros(self.model_dim), margin
        
    def assess_quality(self, z, y_text, x_text, stv_margin):
        with torch.no_grad():
            self.quality_model.eval()
            margin_tensor = torch.tensor([[stv_margin]], dtype=torch.float32)
            quality = self.quality_model(z.unsqueeze(0), y_text, x_text, margin_tensor)
            return quality.item()
            
    def write_lighthouses(self, z_trajectory, y_answer, x_text, task_id, quality):
        """Drop new lighthouse at the final point"""
        # Corrected: pass raw parameters as expected by LighthouseMemory.drop_lighthouse
        lighthouse_id = self.memory.drop_lighthouse(
            vec=z_trajectory,
            b=1.0,
            q=quality,
            y_context=str(y_answer)[:32], # Use a snippet/hash of y_answer
            task_id=task_id
        )
        print(f"Dropped lighthouse {lighthouse_id} with quality {quality:.3f}")
        
    def retrain_quality_model(self):
        if len(self.quality_data) < 10:
            return
        # Simplified retraining for brevity
        pass

def main():
    parser = argparse.ArgumentParser(description="VegaMini Day Loop")
    parser.add_argument("--task", default="arc", help="Task identifier")
    parser.add_argument("--file", help="Input file path")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = VegaMiniRunner()
    
    if args.interactive:
        print("VegaMini Interactive Mode")
        print("Type 'quit' to exit, 'retrain' to retrain quality model")
        
        while True:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            elif query.lower() == 'retrain':
                runner.retrain_quality_model()
                continue
            elif not query:
                continue
                
            try:
                answer, quality = runner.day_step(query, args.task)
                print(f"Answer: {answer}")
                if quality is not None:
                    print(f"Quality: {quality:.3f}")
                else:
                    print("Quality: Below threshold, no lighthouse written")
                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                
    elif args.query:
        answer, quality = runner.day_step(args.query, args.task)
        print(f"Query: {args.query}")
        print(f"Answer: {answer}")
        print(f"Quality: {quality}")
        
    elif args.file:
        # Process file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
            
        with open(file_path) as f:
            if file_path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                else:
                    queries = [data]
            else:
                queries = [line.strip() for line in f if line.strip()]
                
        print(f"Processing {len(queries)} queries from {file_path}")
        
        for i, query in enumerate(queries):
            print(f"\n--- Query {i+1}/{len(queries)} ---")
            try:
                answer, quality = runner.day_step(query, args.task)
                print(f"Answer: {answer}")
                if quality is not None:
                    print(f"Quality: {quality:.3f}")
                    
                # Retrain periodically
                if runner.step_count % 50 == 0:
                    runner.retrain_quality_model()
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
    else:
        print("Please provide --query, --file, or --interactive")

if __name__ == "__main__":
    main()