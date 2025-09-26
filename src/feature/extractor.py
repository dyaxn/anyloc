#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Feature Extractor supporting DINOv2, DINOv3, and ResNet
Backward compatible with original implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Unified feature extractor supporting multiple models
    - DINOv2 (original)
    - DINOv3 
    - ResNet (with pre-trained weights)
    """
    
    def __init__(self, 
                 model_type: str = "dinov2_vitg14",
                 layer: int = 31,
                 facet: str = "value",
                 device: str = "cuda",
                 checkpoint_path: Optional[str] = None):
        """
        Initialize feature extractor
        
        Args:
            model_type: Model type (dinov2_vitg14, dinov3_vitb16, resnet101, etc.)
            layer: Layer index for DINO models
            facet: Facet type for DINO models (query, key, value, token)
            device: Device to use (cuda/cpu)
            checkpoint_path: Path to checkpoint for ResNet models
        """
        self.model_type = model_type.lower()
        self.layer = layer
        self.facet = facet
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Model-specific settings
        self._setup_model_config()
        
        # Load model
        self.model = self._load_model()
        
        # Setup hooks for DINO models
        if "dino" in self.model_type:
            self._setup_hooks()
            
        logger.info(f"Initialized {model_type} extractor on {self.device}")
        
    def _setup_model_config(self):
        """Setup model-specific configurations"""
        if "dinov3" in self.model_type:
            self.patch_size = 16
            self.repo_dir = "/root/zwb/dinov3"  # Configurable
            
            # Model dimensions
            if "vits" in self.model_type:
                self.feat_dim = 384
            elif "vitb" in self.model_type:
                self.feat_dim = 768
            elif "vitl" in self.model_type:
                self.feat_dim = 1024
            elif "vitg" in self.model_type:
                self.feat_dim = 1536
            else:
                self.feat_dim = 768  # default
                
        elif "dinov2" in self.model_type:
            self.patch_size = 14
            
            # Model dimensions
            if "vits" in self.model_type:
                self.feat_dim = 384
            elif "vitb" in self.model_type:
                self.feat_dim = 768
            elif "vitl" in self.model_type:
                self.feat_dim = 1024
            elif "vitg" in self.model_type:
                self.feat_dim = 1536
            else:
                self.feat_dim = 1536  # default for vitg14
                
        elif "resnet" in self.model_type:
            self.patch_size = None  # Not applicable
            self.feat_dim = 2048 if "101" in self.model_type or "152" in self.model_type else 512
            
    def _load_model(self):
        """Load the appropriate model"""
        if "dinov3" in self.model_type:
            return self._load_dinov3()
        elif "dinov2" in self.model_type:
            return self._load_dinov2()
        elif "resnet" in self.model_type:
            return self._load_resnet()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _load_dinov2(self):
        """Load DINOv2 model (original implementation)"""
        logger.info(f"Loading DINOv2 model: {self.model_type}")
        model = torch.hub.load('facebookresearch/dinov2', self.model_type)
        model = model.eval().to(self.device)
        return model
        
    def _load_dinov3(self):
        """Load DINOv3 model"""
        logger.info(f"Loading DINOv3 model: {self.model_type}")
        
        # Map model names to weight paths
        weight_paths = {
            'dinov3_vitb16': "/root/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            'dinov3_vitl16': "/root/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            'dinov3_vits16': "/root/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            'dinov3_vits16plus': "/root/.cache/torch/hub/checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
        }
        
        weight_path = weight_paths.get(self.model_type)
        if not weight_path:
            raise ValueError(f"Unknown DINOv3 model: {self.model_type}")
            
        # Load from local repository
        model = torch.hub.load(
            self.repo_dir,
            self.model_type.replace('dinov3_', ''),  # Remove prefix
            source='local',
            weights=weight_path
        )
        model = model.eval().to(self.device)
        return model
        
    def _load_resnet(self):
        """Load ResNet model"""
        logger.info(f"Loading ResNet model: {self.model_type}")
        
        from torchvision import models
        
        # Load base architecture
        if "resnet50" in self.model_type:
            model = models.resnet50(pretrained=False)
        elif "resnet101" in self.model_type:
            model = models.resnet101(pretrained=False)
        elif "resnet152" in self.model_type:
            model = models.resnet152(pretrained=False)
        else:
            model = models.resnet101(pretrained=False)  # Default
            
        # Remove the last FC layer
        model = nn.Sequential(*list(model.children())[:-2])
        
        # Load checkpoint if provided
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Extract backbone weights only
                state_dict = {}
                for k, v in checkpoint['model_state_dict'].items():
                    if k.startswith('backbone.'):
                        # Remove 'backbone.' prefix
                        new_k = k.replace('backbone.', '')
                        state_dict[new_k] = v
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
        model = model.eval().to(self.device)
        return model
        
    def _setup_hooks(self):
        """Setup hooks for DINO models to extract intermediate features"""
        self._hook_output = None
        
        if "dino" not in self.model_type:
            return
            
        if self.facet == "token":
            # Hook the entire block output
            self.hook = self.model.blocks[self.layer].register_forward_hook(
                lambda m, inp, out: setattr(self, '_hook_output', out))
        else:
            # Hook the QKV output
            self.hook = self.model.blocks[self.layer].attn.qkv.register_forward_hook(
                lambda m, inp, out: setattr(self, '_hook_output', out))
                
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            features: Extracted features
                - For DINO models: [B, N_patches, D]
                - For ResNet: [B, D, H', W'] where H',W' are spatial dims
        """
        images = images.to(self.device)
        
        if "resnet" in self.model_type:
            # ResNet: direct forward pass
            features = self.model(images)
            return features
            
        elif "dino" in self.model_type:
            # DINO: extract from hook
            self._hook_output = None
            _ = self.model(images)
            
            if self._hook_output is None:
                raise RuntimeError(f"Failed to extract features from {self.model_type}")
                
            return self._process_dino_features(self._hook_output)
            
    def _process_dino_features(self, hook_output: torch.Tensor) -> torch.Tensor:
        """Process DINO hook output based on facet type"""
        if self.facet == "token":
            # Direct token output
            features = hook_output
        else:
            # Process QKV output
            B, N, D3 = hook_output.shape
            D = D3 // 3
            
            qkv = hook_output.reshape(B, N, 3, D)
            
            if self.facet == "query":
                features = qkv[:, :, 0, :]
            elif self.facet == "key":
                features = qkv[:, :, 1, :]
            elif self.facet == "value":
                features = qkv[:, :, 2, :]
            else:
                raise ValueError(f"Unknown facet: {self.facet}")
                
        # Remove CLS token (first token)
        features = features[:, 1:, :]
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=-1)
        
        return features
        
    def get_patch_size(self) -> Optional[int]:
        """Get patch size for the model"""
        return self.patch_size
        
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return self.feat_dim
        
    def __repr__(self):
        return f"FeatureExtractor(model={self.model_type}, layer={self.layer}, facet={self.facet})"


# Backward compatibility classes
class DinoV2Extractor(FeatureExtractor):
    """Backward compatible DINOv2 extractor"""
    def __init__(self, model_name="dinov2_vitg14", layer=31, facet="value", device="cuda"):
        super().__init__(model_type=model_name, layer=layer, facet=facet, device=device)


class DinoV3Extractor(FeatureExtractor):
    """DINOv3 specific extractor"""
    def __init__(self, model_name="dinov3_vitb16", layer=11, facet="value", device="cuda"):
        super().__init__(model_type=model_name, layer=layer, facet=facet, device=device)


class ResNetExtractor(FeatureExtractor):
    """ResNet specific extractor"""
    def __init__(self, model_name="resnet101", checkpoint_path=None, device="cuda"):
        super().__init__(model_type=model_name, checkpoint_path=checkpoint_path, device=device)