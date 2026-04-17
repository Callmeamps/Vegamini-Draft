import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import hashlib

def cluster_y_candidates(y_candidates: List[torch.Tensor], n_clusters: int = None) -> List[List[int]]:
    """Cluster y candidates by similarity."""
    if len(y_candidates) <= 1:
        return [[0]] if y_candidates else []
        
    # Stack candidates for clustering
    y_stacked = torch.stack(y_candidates).cpu().numpy()
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = min(max(2, len(y_candidates) // 4), 8)
    
    n_clusters = min(n_clusters, len(y_candidates))
    
    # K-means clustering
    if n_clusters == 1:
        return [list(range(len(y_candidates)))]
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(y_stacked)
    
    # Group indices by cluster
    clusters = []
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if cluster_indices:  # Only add non-empty clusters
            clusters.append(cluster_indices)
    
    return clusters

def rank_clusters_by_worker(clusters: List[List[int]], y_candidates: List[torch.Tensor], 
                          z_trajectories: List[torch.Tensor]) -> List[List[int]]:
    """Each worker ranks clusters by their internal consistency."""
    
    def cluster_score(cluster_indices: List[int]) -> float:
        if len(cluster_indices) <= 1:
            return 0.0
            
        # Score based on internal consistency of y vectors
        cluster_ys = [y_candidates[i] for i in cluster_indices]
        y_stack = torch.stack(cluster_ys)
        
        # Compute pairwise similarities
        similarities = torch.cosine_similarity(y_stack.unsqueeze(1), y_stack.unsqueeze(0), dim=-1)
        
        # Exclude diagonal and take mean
        mask = ~torch.eye(len(cluster_ys), dtype=torch.bool)
        mean_similarity = similarities[mask].mean().item()
        
        return mean_similarity
    
    # Score all clusters
    cluster_scores = [(i, cluster_score(cluster)) for i, cluster in enumerate(clusters)]
    
    # Sort by score (descending)
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return ranked cluster indices
    return [cluster_scores[i][0] for i in range(len(cluster_scores))]

def single_transferable_vote(ballots: List[List[int]], clusters: List[List[int]]) -> Tuple[int, float]:
    """
    Simplified STV implementation.
    Returns (winning_cluster_index, victory_margin)
    """
    if not ballots or not clusters:
        return 0, 0.0
        
    n_clusters = len(clusters)
    if n_clusters == 1:
        return 0, 1.0
        
    # Count first preferences
    votes = [0] * n_clusters
    for ballot in ballots:
        if ballot:
            votes[ballot[0]] += 1
            
    total_votes = sum(votes)
    if total_votes == 0:
        return 0, 0.0
        
    # Find winner and margin
    winner_idx = max(range(n_clusters), key=lambda i: votes[i])
    winner_votes = votes[winner_idx]
    
    # Calculate margin (lead over second place)
    sorted_votes = sorted(votes, reverse=True)
    margin = (sorted_votes[0] - sorted_votes[1]) / total_votes if len(sorted_votes) > 1 else 1.0
    
    return winner_idx, margin

def hash_y_context(y: torch.Tensor) -> str:
    """Create a hash for y context matching."""
    y_bytes = y.detach().cpu().numpy().astype(np.float32).tobytes()
    return hashlib.md5(y_bytes).hexdigest()[:16]