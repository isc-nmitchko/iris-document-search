import numpy as np

import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings

class MUVERAEncoder:
    """
    MUVERA (Multi-Vector Retrieval Algorithm) Fixed Dimensional Encoding implementation.
    
    Based on the paper: "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
    https://arxiv.org/pdf/2405.19504
    
    This implementation converts multi-vector embeddings into single fixed-dimensional vectors
    whose dot product approximates the Chamfer similarity between multi-vector sets.
    """
    
    def __init__(
        self,
        k_sim: int = 4,
        d_proj: Optional[int] = None,
        R_reps: int = 5,
        d_final: Optional[int] = None,
        fill_empty_clusters: bool = True,
        seed: int = 42
    ):
        """
        Initialize MUVERA encoder.
        
        Args:
            k_sim: Number of SimHash hyperplanes (creates 2^k_sim clusters)
            d_proj: Projection dimension for each block (None = no projection)
            R_reps: Number of repetitions for better approximation
            d_final: Final projection dimension (None = no final projection)
            fill_empty_clusters: Whether to fill empty clusters for document FDEs
            seed: Random seed for reproducibility
        """
        self.k_sim = k_sim
        self.d_proj = d_proj
        self.R_reps = R_reps
        self.d_final = d_final
        self.fill_empty_clusters = fill_empty_clusters
        self.seed = seed
        
        # Number of clusters per repetition
        self.B = 2 ** k_sim
        
        # Store random matrices for reproducibility
        self.gaussian_matrices = []
        self.projection_matrices = []
        self.final_projection_matrix = None
        
        self._initialize_random_matrices()
    
    def _initialize_random_matrices(self):
        """Initialize all random matrices used in the encoding process."""
        np.random.seed(self.seed)
        
        # For each repetition, create SimHash Gaussian vectors and projection matrices
        for rep in range(self.R_reps):
            # SimHash Gaussian vectors: k_sim vectors of dimension d (will be set later)
            self.gaussian_matrices.append(None)  # Will be initialized when we know d
            
            # Projection matrices (if d_proj is specified)
            self.projection_matrices.append(None)  # Will be initialized when we know d
    
    def _initialize_matrices_for_dimension(self, d: int):
        """Initialize dimension-dependent matrices."""
        if self.gaussian_matrices[0] is not None:
            return  # Already initialized
            
        np.random.seed(self.seed)
        
        for rep in range(self.R_reps):
            # SimHash Gaussian vectors: k_sim × d matrix
            self.gaussian_matrices[rep] = np.random.randn(self.k_sim, d)
            
            # Projection matrices (random ±1 matrices)
            if self.d_proj is not None and self.d_proj < d:
                proj_matrix = np.random.choice([-1, 1], size=(self.d_proj, d))
                self.projection_matrices[rep] = proj_matrix / np.sqrt(self.d_proj)
            else:
                self.projection_matrices[rep] = None
        
        # Final projection matrix
        if self.d_final is not None:
            fde_dim = self.get_fde_dimension(d)
            if self.d_final < fde_dim:
                self.final_projection_matrix = np.random.choice(
                    [-1, 1], size=(self.d_final, fde_dim)
                ) / np.sqrt(self.d_final)
    
    def get_fde_dimension(self, d: int) -> int:
        """Calculate the FDE dimension for given input dimension."""
        block_dim = self.d_proj if self.d_proj is not None else d
        return self.B * block_dim * self.R_reps
    
    def _sim_hash_partition(self, vectors: np.ndarray, rep: int) -> np.ndarray:
        """
        Apply SimHash partitioning to vectors.
        
        Args:
            vectors: Array of shape (n, d)
            rep: Repetition index
            
        Returns:
            Partition assignments as integers (n,)
        """
        # Compute dot products with Gaussian vectors
        dots = np.dot(vectors, self.gaussian_matrices[rep].T)  # (n, k_sim)
        
        # Convert to binary (sign function)
        binary = (dots > 0).astype(int)  # (n, k_sim)
        
        # Convert binary arrays to integers
        powers = 2 ** np.arange(self.k_sim)
        partitions = np.dot(binary, powers)  # (n,)
        
        return partitions
    
    def _apply_projection(self, vector: np.ndarray, rep: int) -> np.ndarray:
        """Apply random projection if specified."""
        if self.projection_matrices[rep] is not None:
            return np.dot(self.projection_matrices[rep], vector)
        return vector
    
    def _hamming_distance(self, a: int, b: int) -> int:
        """Calculate Hamming distance between two integers (as bit strings)."""
        return bin(a ^ b).count('1')
    
    def encode_query(self, multi_vecs: np.ndarray) -> np.ndarray:
        """
        Encode query multi-vectors into FDE.
        
        Args:
            multi_vecs: Query vectors of shape (M, d)
            
        Returns:
            FDE vector of shape (fde_dim,)
        """
        M, d = multi_vecs.shape
        self._initialize_matrices_for_dimension(d)
        
        # Normalize vectors (as assumed in the paper)
        multi_vecs = multi_vecs / (np.linalg.norm(multi_vecs, axis=1, keepdims=True) + 1e-8)
        
        fde_parts = []
        
        for rep in range(self.R_reps):
            # Get partition assignments
            partitions = self._sim_hash_partition(multi_vecs, rep)
            
            # Initialize blocks for this repetition
            block_dim = self.d_proj if self.d_proj is not None else d
            rep_fde = np.zeros(self.B * block_dim)
            
            # For each cluster, sum the vectors that belong to it
            for cluster in range(self.B):
                mask = (partitions == cluster)
                if np.any(mask):
                    # Sum vectors in this cluster
                    cluster_sum = np.sum(multi_vecs[mask], axis=0)
                    
                    # Apply projection if specified
                    cluster_sum_proj = self._apply_projection(cluster_sum, rep)
                    
                    # Store in the appropriate block
                    start_idx = cluster * len(cluster_sum_proj)
                    end_idx = start_idx + len(cluster_sum_proj)
                    rep_fde[start_idx:end_idx] = cluster_sum_proj
            
            fde_parts.append(rep_fde)
        
        # Concatenate all repetitions
        fde = np.concatenate(fde_parts)
        
        # Apply final projection if specified
        if self.final_projection_matrix is not None:
            fde = np.dot(self.final_projection_matrix, fde)
        
        # Normalize the final FDE
        norm = np.linalg.norm(fde)
        if norm > 0:
            fde = fde / norm
            
        return fde
    
    def encode_document(self, multi_vecs: np.ndarray) -> np.ndarray:
        """
        Encode document multi-vectors into FDE.
        
        Args:
            multi_vecs: Document vectors of shape (M, d)
            
        Returns:
            FDE vector of shape (fde_dim,)
        """
        M, d = multi_vecs.shape
        self._initialize_matrices_for_dimension(d)
        
        # Normalize vectors (as assumed in the paper)
        multi_vecs = multi_vecs / (np.linalg.norm(multi_vecs, axis=1, keepdims=True) + 1e-8)
        
        fde_parts = []
        
        for rep in range(self.R_reps):
            # Get partition assignments
            partitions = self._sim_hash_partition(multi_vecs, rep)
            
            # Initialize blocks for this repetition
            block_dim = self.d_proj if self.d_proj is not None else d
            rep_fde = np.zeros(self.B * block_dim)
            
            # Track which clusters are empty for fill_empty_clusters
            occupied_clusters = set(partitions)
            
            # For each cluster, compute the centroid of vectors that belong to it
            for cluster in range(self.B):
                mask = (partitions == cluster)
                
                if np.any(mask):
                    # Compute centroid (mean) of vectors in this cluster
                    cluster_centroid = np.mean(multi_vecs[mask], axis=0)
                else:
                    # Handle empty cluster
                    if self.fill_empty_clusters:
                        # Find the closest point to this cluster
                        min_distance = float('inf')
                        closest_point = None
                        
                        for i, vec in enumerate(multi_vecs):
                            distance = self._hamming_distance(partitions[i], cluster)
                            if distance < min_distance:
                                min_distance = distance
                                closest_point = vec
                        
                        cluster_centroid = closest_point if closest_point is not None else np.zeros(d)
                    else:
                        cluster_centroid = np.zeros(d)
                
                # Apply projection if specified
                cluster_centroid_proj = self._apply_projection(cluster_centroid, rep)
                
                # Store in the appropriate block
                start_idx = cluster * len(cluster_centroid_proj)
                end_idx = start_idx + len(cluster_centroid_proj)
                rep_fde[start_idx:end_idx] = cluster_centroid_proj
            
            fde_parts.append(rep_fde)
        
        # Concatenate all repetitions
        fde = np.concatenate(fde_parts)
        
        # Apply final projection if specified
        if self.final_projection_matrix is not None:
            fde = np.dot(self.final_projection_matrix, fde)
        
        # Normalize the final FDE
        norm = np.linalg.norm(fde)
        if norm > 0:
            fde = fde / norm
            
        return fde
    
    def chamfer_similarity(self, Q: np.ndarray, P: np.ndarray) -> float:
        """
        Compute exact Chamfer similarity for comparison.
        
        Args:
            Q: Query vectors (M1, d)
            P: Document vectors (M2, d)
            
        Returns:
            Chamfer similarity value
        """
        # Normalize vectors
        Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        P_norm = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        sim_matrix = np.dot(Q_norm, P_norm.T)  # (M1, M2)
        
        # For each query vector, find max similarity with any document vector
        max_sims = np.max(sim_matrix, axis=1)  # (M1,)
        
        # Sum over all query vectors
        return np.sum(max_sims)


def reduce_dimensions(vector, target_dim=10000):
    truncated_fde = vector[:target_dim]  # First 10000 elements
    truncated_fde /= np.linalg.norm(truncated_fde)
    return truncated_fde
