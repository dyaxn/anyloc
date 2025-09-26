#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过模型和聚合器选择增强的特征提取脚本
实现向后兼容
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature.extractor import FeatureExtractor
from src.feature.aggregator import FeatureAggregator
from src.feature.preprocessor import ImagePreprocessor
from src.utils.logger import setup_logger

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logger
logger = setup_logger('extract_features')


def load_config(config_path: str = None) -> dict:
    """Load configuration file"""
    if config_path is None:
        config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_image_paths(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """Get all image paths from directory"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f'*{ext}'))
        image_paths.extend(directory.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    return image_paths


def extract_features_batch(
    image_paths: list,
    extractor: FeatureExtractor,
    aggregator: FeatureAggregator,
    preprocessor: ImagePreprocessor,
    batch_size: int = 32,
    device: str = 'cuda'
) -> tuple:
    """
    Extract features for a batch of images
    
    Returns:
        descriptors: numpy array of shape [N, D]
        image_ids: list of image identifiers
    """
    all_descriptors = []
    all_image_ids = []
    
    # Process in batches
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
        # Get batch paths
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        batch_tensors = []
        batch_ids = []
        
        # Preprocess images
        for img_path in batch_paths:
            try:
                img_tensor = preprocessor.process(img_path)
                batch_tensors.append(img_tensor)
                batch_ids.append(img_path.stem)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue
        
        if len(batch_tensors) == 0:
            continue
        
        # Stack into batch
        batch = torch.stack(batch_tensors)
        
        # Extract features
        with torch.no_grad():
            # Extract base features
            features = extractor.extract(batch)
            
            # Aggregate features
            if aggregator.agg_type == "vlad":
                # VLAD needs per-image processing
                descriptors = []
                for i in range(features.shape[0]):
                    desc = aggregator.aggregate(features[i])
                    descriptors.append(desc.cpu().numpy())
                descriptors = np.array(descriptors)
            else:
                # GeM can handle batch
                descriptors = aggregator.aggregate(features).cpu().numpy()
            
            all_descriptors.append(descriptors)
            all_image_ids.extend(batch_ids)
        
        # Clear GPU cache periodically
        if device == 'cuda' and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all descriptors
    if len(all_descriptors) > 0:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    
    return all_descriptors, all_image_ids


def save_descriptors(
    descriptors: np.ndarray,
    image_ids: list,
    save_path: Path,
    metadata: dict = None
) -> None:
    """Save descriptors with metadata"""
    
    # Prepare save data
    save_data = {
        'descriptors': torch.from_numpy(descriptors),
        'image_ids': image_ids,
        'shape': descriptors.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        save_data['metadata'] = metadata
    
    # Save as PyTorch file
    torch.save(save_data, save_path)
    
    # Also save metadata as JSON
    metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_images': len(image_ids),
            'descriptor_dim': int(descriptors.shape[1]),
            'timestamp': save_data['timestamp'],
            'metadata': metadata
        }, f, indent=2)
    
    logger.info(f"Saved descriptors to {save_path}")
    logger.info(f"  Shape: {descriptors.shape}")
    logger.info(f"  File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract features from images")
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for descriptors')
    parser.add_argument('--mode', type=str, default='database',
                       choices=['database', 'query'],
                       help='Processing mode')
    
    # Model selection
    parser.add_argument('--model', type=str, default=None,
                       help='Model type (e.g., dinov2_vitg14, dinov3_vitb16, resnet101)')
    parser.add_argument('--layer', type=int, default=None,
                       help='Layer index for DINO models')
    parser.add_argument('--facet', type=str, default=None,
                       choices=['query', 'key', 'value', 'token'],
                       help='Facet for DINO models')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for ResNet models')
    
    # Aggregator selection
    parser.add_argument('--agg', type=str, default=None,
                       choices=['vlad', 'gem'],
                       help='Aggregation method')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Vocabulary path for VLAD')
    parser.add_argument('--gem_p', type=float, default=None,
                       help='GeM pooling parameter')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--profile', type=str, default=None,
                       help='Use predefined profile from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply profile if specified
    if args.profile and args.profile in config.get('profiles', {}):
        profile = config['profiles'][args.profile]
        logger.info(f"Using profile: {args.profile}")
        
        # Update config with profile settings
        if 'extractor' in profile:
            config['model']['extractor'].update(profile['extractor'])
        if 'aggregator' in profile:
            config['model']['aggregator'].update(profile['aggregator'])
    
    # Override with command line arguments
    if args.model:
        config['model']['extractor']['type'] = args.model
    if args.layer:
        config['model']['extractor']['layer'] = args.layer
    if args.facet:
        config['model']['extractor']['facet'] = args.facet
    if args.checkpoint:
        config['model']['extractor']['checkpoint_path'] = args.checkpoint
    
    if args.agg:
        config['model']['aggregator']['type'] = args.agg
    if args.vocab:
        config['model']['aggregator']['vlad']['vocab_path'] = args.vocab
    if args.gem_p:
        config['model']['aggregator']['gem']['p'] = args.gem_p
    
    # Set input/output directories
    if args.mode == 'database':
        input_dir = Path(args.input_dir or config['data']['database_dir'])
        output_name = 'database_descriptors'
    else:
        input_dir = Path(args.input_dir or config['data']['query_dir'])
        output_name = 'query_descriptors'
    
    output_dir = Path(args.output_dir or config['data']['descriptor_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model and aggregator settings
    model_type = config['model']['extractor']['type']
    agg_type = config['model']['aggregator']['type']
    
    # Create output filename
    model_suffix = model_type.replace('/', '_')
    agg_suffix = agg_type
    output_file = output_dir / f"{output_name}_{model_suffix}_{agg_suffix}.pt"
    
    logger.info("="*60)
    logger.info("Feature Extraction Configuration")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {model_type}")
    logger.info(f"Aggregator: {agg_type}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_file}")
    logger.info("="*60)
    
    # Get image paths
    image_paths = get_image_paths(input_dir)
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        logger.error("No images found!")
        return
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Extractor
    extractor = FeatureExtractor(
        model_type=model_type,
        layer=config['model']['extractor'].get('layer', 31),
        facet=config['model']['extractor'].get('facet', 'value'),
        device=args.device,
        checkpoint_path=config['model']['extractor'].get('checkpoint_path')
    )
    
    # Aggregator
    if agg_type == 'vlad':
        # Get vocabulary path
        vocab_path = config['model']['aggregator']['vlad'].get('vocab_path')
        
        # Check for model-specific vocabulary
        vocab_paths = config['model']['aggregator']['vlad'].get('vocab_paths', {})
        if model_type in vocab_paths:
            vocab_path = vocab_paths[model_type]
        
        aggregator = FeatureAggregator(
            agg_type='vlad',
            vocab_path=vocab_path,
            device=args.device
        )
    else:
        aggregator = FeatureAggregator(
            agg_type='gem',
            gem_p=config['model']['aggregator']['gem'].get('p', 3.0),
            device=args.device
        )
    
    # Preprocessor
    preprocessor = ImagePreprocessor(
        target_size=None,  # Will be auto-determined
        mode=args.mode,
        patch_size=extractor.get_patch_size()
    )
    
    # Extract features
    logger.info("Starting feature extraction...")
    descriptors, image_ids = extract_features_batch(
        image_paths,
        extractor,
        aggregator,
        preprocessor,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Prepare metadata
    metadata = {
        'mode': args.mode,
        'model': {
            'type': model_type,
            'layer': config['model']['extractor'].get('layer'),
            'facet': config['model']['extractor'].get('facet')
        },
        'aggregator': {
            'type': agg_type,
            'params': {
                'vocab_path': config['model']['aggregator']['vlad'].get('vocab_path') if agg_type == 'vlad' else None,
                'gem_p': config['model']['aggregator']['gem'].get('p') if agg_type == 'gem' else None
            }
        },
        'preprocessing': {
            'patch_size': extractor.get_patch_size()
        }
    }
    
    # Save descriptors
    save_descriptors(descriptors, image_ids, output_file, metadata)
    
    # Print statistics
    logger.info("="*60)
    logger.info("Extraction Complete!")
    logger.info(f"  Total images: {len(image_ids)}")
    logger.info(f"  Descriptor dimension: {descriptors.shape[1]}")
    logger.info(f"  Average norm: {np.linalg.norm(descriptors, axis=1).mean():.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()