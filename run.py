import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path


from vega_mini.memory.punk import LighthouseMemory
from vega_mini.memory.lighthouse import Lighthouse
from vega_mini.controller.trm import VegaMiniTransformer
from vega_mini.controller.flow import FlowSolver
from vega_mini.eval.quality import QualityModel


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
        
        Args:
            x_text: Input text/question
            task_id: Task identifier
            
        Returns:
            tuple: (y_answer, quality_score)
        """
        # If input is a dict (e.g., {'input': '...'}), extract the string
        if isinstance(x_text, dict) and 'input' in x_text:
            x_text = x_text['input']
        # If input is not a string, convert to string
        if not isinstance(x_text, str):
            x_text = str(x_text)
        print(f"Day step {self.step_count}: Processing '{x_text[:50]}...'")


        # Embed x_text as a tensor for model input
        x_embed = self.embed_text(x_text, dim=self.model_dim)

        # Get live anchors for this task
        anchors = self.memory.get_live_anchors(task_id, top_k=64)
        print(f"Found {len(anchors)} live anchors")

        # Swarm: generate candidates
        y_candidates = []
        z_trajectories = []

        swarm_size = min(32, max(4, len(anchors) // 2))  # Adaptive swarm size
        print(f"Running swarm with {swarm_size} workers")

        for worker_id in range(swarm_size):
            # Start from random point
            z0 = torch.randn(self.model_dim)
            # Flow solve with anchors, using embedded x_text
            z_final, _ = self.flow_solver.solve_flow(
                z0, self.controller.velocity_net, x_embed, y=None, anchors=anchors, t_steps=6
            )
            # Generate answer using controller (assume y_proj for now)
            y_answer = self.controller.y_proj(z_final.unsqueeze(0)).squeeze(0)
            y_candidates.append(y_answer)
            z_trajectories.append(z_final)
            
        # STV voting on clusters
        print("Running STV voting...")
        clusters = self.cluster_answers(y_candidates)
        ballots = self.rank_by_worker(clusters, swarm_size)
        y_winner, stv_margin = self.stv_vote(ballots, y_candidates)
        # Find winning trajectory
        try:
            win_idx = y_candidates.index(y_winner)
        except ValueError:
            # If y_winner is a tensor, use tensor comparison
            win_idx = next((i for i, y in enumerate(y_candidates) if torch.allclose(y, y_winner)), 0)
        z_winner = z_trajectories[win_idx]
        
        # Quality assessment
        quality_score = self.assess_quality(z_winner, y_winner, x_text, stv_margin)
        print(f"Quality score: {quality_score:.3f}, STV margin: {stv_margin:.3f}")
        
        # Write lighthouses if quality is high
        if quality_score > 0.8:
            self.write_lighthouses(z_winner, y_winner, x_text, task_id, quality_score)
        # Reinforce nearby anchors
        if quality_score > 0.65:
            self.memory.reinforce_nearby(z_winner, delta_b=0.1 * quality_score)
            
        # Store training data for quality model
        self.quality_data.append({
            'z': z_winner.detach().clone(),
            'y': y_winner,
            'x': x_text,
            'stv_margin': stv_margin,
            'quality': quality_score
        })
        
        self.step_count += 1
        return y_winner, quality_score if quality_score > 0.65 else None
        
    def cluster_answers(self, y_candidates, similarity_threshold=0.9):
        """Group similar answers into clusters using cosine similarity."""
        if not y_candidates:
            return []
        # Assume y_candidates are tensors (embeddings)
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
        """Each worker ranks the clusters"""
        ballots = []
        for worker_id in range(num_workers):
            # Simple ranking: prefer clusters with more members
            ranked = sorted(clusters, key=len, reverse=True)
            ballots.append(ranked)
        return ballots
        
    def stv_vote(self, ballots, y_candidates=None):
        """Single Transferable Vote on answer clusters. Returns winning answer and margin."""
        if not ballots or not ballots[0]:
            return "No answer", 0.0
        # Flatten all clusters from ballots
        all_clusters = [cluster for ballot in ballots for cluster in ballot]
        if not all_clusters:
            return "No answer", 0.0
        # Find the most common cluster (by index list)
        from collections import Counter
        cluster_tuples = [tuple(sorted(cluster)) for cluster in all_clusters]
        most_common = Counter(cluster_tuples).most_common(1)
        if not most_common:
            return "No answer", 0.0
        winner_indices = list(most_common[0][0])
        margin = len(winner_indices) / max(sum(len(cluster) for cluster in all_clusters), 1)
        # Return the first answer in the winning cluster (if y_candidates provided)
        if y_candidates and winner_indices:
            return y_candidates[winner_indices[0]], margin
        return f"cluster_{winner_indices[0]}", margin
        
    def assess_quality(self, z, y_text, x_text, stv_margin):
        """Use quality model to score the answer"""
        with torch.no_grad():
            self.quality_model.eval()
            margin_tensor = torch.tensor([[stv_margin]], dtype=torch.float32)
            quality = self.quality_model(z.unsqueeze(0), y_text, x_text, margin_tensor)
            return quality.item()
            
    def write_lighthouses(self, z_trajectory, y_answer, x_text, task_id, quality):
        """Drop new lighthouses along the winning trajectory"""
        # For now, just drop one lighthouse at the final point
        lighthouse = Lighthouse(
            vec=z_trajectory.detach().numpy(),
            brightness=1.0,
            quality=quality,
            y_context=y_answer,
            task_id=task_id
        )
        
        lighthouse_id = self.memory.drop_lighthouse(lighthouse)
        print(f"Dropped lighthouse {lighthouse_id} with quality {quality:.3f}")
        
    def retrain_quality_model(self):
        """Retrain quality model on collected data"""
        if len(self.quality_data) < 10:
            return
            
        print(f"Retraining quality model on {len(self.quality_data)} samples...")
        
        # Prepare training data
        z_batch = torch.stack([d['z'] for d in self.quality_data])
        stv_batch = torch.tensor([[d['stv_margin']] for d in self.quality_data], dtype=torch.float32)
        target_batch = torch.tensor([d['quality'] for d in self.quality_data], dtype=torch.float32)
        
        # Train for a few steps
        optimizer = torch.optim.Adam(self.quality_model.parameters(), lr=1e-3)
        self.quality_model.train()
        
        for _ in range(10):
            for i in range(0, len(z_batch), 8):  # Mini-batches of 8
                end_idx = min(i + 8, len(z_batch))
                z_mini = z_batch[i:end_idx]
                stv_mini = stv_batch[i:end_idx]
                target_mini = target_batch[i:end_idx]
                
                # Use first sample's text for batch (simplification)
                y_text = self.quality_data[i]['y']
                x_text = self.quality_data[i]['x']
                
                loss = self.quality_model.train_step(
                    z_mini, y_text, x_text, stv_mini, target_mini, optimizer
                )
                
        print("Quality model retraining complete")

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