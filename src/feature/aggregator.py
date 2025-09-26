#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Feature Aggregator supporting VLAD and GeM pooling
Backward compatible with original VLAD implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class FeatureAggregator:
    """
    Unified feature aggregator supporting multiple methods
    - VLAD (Vector of Locally Aggregated Descriptors)
    - GeM (Generalized Mean) pooling
    """
    
    def __init__(self,
                 agg_type: str = "vlad",
                 vocab_path: Optional[str] = None,
                 num_clusters: int = 64,
                 gem_p: float = 3.0,
                 device: str = "cuda"):
        """
        Initialize feature aggregator
        
        Args:
            agg_type: Aggregation type ('vlad' or 'gem')
            vocab_path: Path to VLAD vocabulary (required for VLAD)
            num_clusters: Number of clusters for VLAD
            gem_p: Power parameter for GeM pooling
            device: Device to use
        """
        self.agg_type = agg_type.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if self.agg_type == "vlad":
            if vocab_path is None:
                raise ValueError("vocab_path is required for VLAD aggregation")
            self.vocab_path = vocab_path
            self.num_clusters = num_clusters
            self._load_vlad_vocabulary()
            
        elif self.agg_type == "gem":
            self.gem_p = gem_p
            logger.info(f"Using GeM pooling with p={gem_p}")
            
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")
            
        logger.info(f"Initialized {agg_type} aggregator")
        
    def _load_vlad_vocabulary(self):
        """Load VLAD vocabulary/cluster centers"""
        if not Path(self.vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
            
        logger.info(f"Loading VLAD vocabulary from {self.vocab_path}")
        
        # Load vocabulary (support multiple formats)
        data = torch.load(self.vocab_path, map_location='cpu')
        
        if isinstance(data, dict):
            # New format with metadata
            if 'c_centers' in data:
                self.c_centers = data['c_centers'].to(self.device)
            elif 'centers' in data:
                self.c_centers = data['centers'].to(self.device)
            elif 'centroids' in data:
                self.c_centers = data['centroids'].to(self.device)
            else:
                raise KeyError("Cannot find cluster centers in vocabulary file")
                
            # Load metadata if available
            if 'config' in data:
                vocab_config = data['config']
                logger.info(f"Vocabulary info: {vocab_config}")
                
        else:
            # Old format: direct tensor
            self.c_centers = data.to(self.device)
            
        self.num_clusters, self.desc_dim = self.c_centers.shape
        logger.info(f"Loaded {self.num_clusters} clusters of dimension {self.desc_dim}")
        
    @torch.no_grad()
    def aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features using the specified method
        
        Args:
            features: Input features
                - For DINO models: [B, N_patches, D] or [N_patches, D]
                - For ResNet: [B, D, H, W] or [D, H, W]
                
        Returns:
            aggregated: Aggregated features
                - For VLAD: [B, num_clusters * D] or [num_clusters * D]
                - For GeM: [B, D] or [D]
        """
        if self.agg_type == "vlad":
            return self._vlad_aggregate(features)
        elif self.agg_type == "gem":
            return self._gem_aggregate(features)
            
    def _vlad_aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """
        VLAD aggregation
        
        Args:
            features: [B, N_patches, D] or [N_patches, D]
            
        Returns:
            vlad: [B, num_clusters * D] or [num_clusters * D]
        """
        features = features.to(self.device)
        
        # Handle both batched and single inputs
        if features.dim() == 2:
            # Single image: [N_patches, D]
            return self._vlad_single(features)
        elif features.dim() == 3:
            # Batch: [B, N_patches, D]
            batch_size = features.shape[0]
            vlad_list = []
            for i in range(batch_size):
                vlad_single = self._vlad_single(features[i])
                vlad_list.append(vlad_single)
            return torch.stack(vlad_list, dim=0)
        else:
            raise ValueError(f"Unexpected feature dimensions: {features.shape}")
            
    def _vlad_single(self, desc: torch.Tensor) -> torch.Tensor:
        """
        VLAD for single image
        
        Args:
            desc: [N_patches, D]
            
        Returns:
            vlad: [num_clusters * D]
        """
        N, D = desc.shape
        
        # Compute distances to all clusters
        dists = torch.cdist(desc, self.c_centers, p=2)  # [N, K]
        
        # Hard assignment
        assign = torch.argmin(dists, dim=1)  # [N]
        
        # Compute residuals and aggregate
        vlad = torch.zeros(self.num_clusters, D, device=self.device)
        
        for k in range(self.num_clusters):
            mask = (assign == k)
            if torch.any(mask):
                # Residuals for patches assigned to cluster k
                residuals = desc[mask] - self.c_centers[k].unsqueeze(0)
                # Sum residuals
                vlad[k] = residuals.sum(dim=0)
                
        # Intra-normalization (normalize each cluster's residuals)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten
        vlad = vlad.flatten()
        
        # Final L2 normalization
        vlad = F.normalize(vlad, p=2, dim=0)
        
        return vlad
        
    def _gem_aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generalized Mean (GeM) pooling
        
        Args:
            features: 
                - For DINO: [B, N_patches, D] or [N_patches, D]
                - For ResNet: [B, D, H, W] or [D, H, W]
                
        Returns:
            pooled: [B, D] or [D]
        """
        features = features.to(self.device)
        eps = 1e-6
        
        # Handle different input formats
        if features.dim() == 2:
            # Single DINO: [N_patches, D]
            features = torch.clamp(features, min=eps)
            pooled = torch.mean(features.pow(self.gem_p), dim=0).pow(1.0/self.gem_p)
            pooled = F.normalize(pooled, p=2, dim=0)
            return pooled
            
        elif features.dim() == 3:
            # Batch DINO: [B, N_patches, D]
            features = torch.clamp(features, min=eps)
            pooled = torch.mean(features.pow(self.gem_p), dim=1).pow(1.0/self.gem_p)
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled
            
        elif features.dim() == 4:
            # ResNet features: [B, C, H, W]
            B, C, H, W = features.shape
            features = torch.clamp(features, min=eps)
            
            # Apply GeM spatially
            features = features.pow(self.gem_p)
            pooled = F.adaptive_avg_pool2d(features, (1, 1))
            pooled = pooled.pow(1.0/self.gem_p)
            pooled = pooled.view(B, C)
            
            # L2 normalize
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled
            
        else:
            raise ValueError(f"Unexpected feature dimensions: {features.shape}")
            
    def get_output_dim(self) -> int:
        """Get output dimension after aggregation"""
        if self.agg_type == "vlad":
            return self.num_clusters * self.desc_dim
        elif self.agg_type == "gem":
            # GeM preserves the feature dimension
            return None  # Depends on input
        else:
            return None
            
    def __repr__(self):
        if self.agg_type == "vlad":
            return f"FeatureAggregator(type=VLAD, clusters={self.num_clusters}, dim={self.desc_dim})"
        elif self.agg_type == "gem":
            return f"FeatureAggregator(type=GeM, p={self.gem_p})"
        else:
            return f"FeatureAggregator(type={self.agg_type})"


# Backward compatibility classes
class VLADAggregator(FeatureAggregator):
    """Backward compatible VLAD aggregator"""
    def __init__(self, vocab_path: str, num_clusters: int = 64, device: str = "cuda"):
        super().__init__(agg_type="vlad", vocab_path=vocab_path, 
                        num_clusters=num_clusters, device=device)
        
    def aggregate_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Legacy method name"""
        return self.aggregate(patches)


class GeMPooling(nn.Module):
    """GeM pooling as a PyTorch module (for training)"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = torch.tensor([p])
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling
        
        Args:
            x: [B, C, H, W] for CNN or [B, N, D] for transformers
            
        Returns:
            pooled: [B, C] or [B, D]
        """
        # Clamp p value to reasonable range
        p = torch.clamp(self.p, min=self.eps, max=10.0)
        
        if x.dim() == 4:
            # CNN features: [B, C, H, W]
            x = torch.clamp(x, min=self.eps)
            x = x.pow(p)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.pow(1.0 / p)
            return x.view(x.size(0), -1)
            
        elif x.dim() == 3:
            # Transformer features: [B, N, D]
            x = torch.clamp(x, min=self.eps)
            x = x.pow(p)
            x = torch.mean(x, dim=1)
            x = x.pow(1.0 / p)
            return x
            
        else:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
            
    def __repr__(self):
        return f"GeMPooling(p={self.p.item() if hasattr(self.p, 'item') else self.p})"


def create_aggregator(config: dict) -> FeatureAggregator:
    """
    Factory function to create aggregator from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        aggregator: FeatureAggregator instance
    """
    agg_type = config.get('type', 'vlad')
    
    if agg_type == 'vlad':
        return FeatureAggregator(
            agg_type='vlad',
            vocab_path=config.get('vocab_path'),
            num_clusters=config.get('num_clusters', 64),
            device=config.get('device', 'cuda')
        )
    elif agg_type == 'gem':
        return FeatureAggregator(
            agg_type='gem',
            gem_p=config.get('gem_p', 3.0),
            device=config.get('device', 'cuda')
        )
    else:
        raise ValueError(f"Unknown aggregator type: {agg_type}")