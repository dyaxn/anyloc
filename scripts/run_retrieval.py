#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
具有模型和聚合器选择的增强检索脚本
实现向后兼容
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import time
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature.extractor import FeatureExtractor
from src.feature.aggregator import FeatureAggregator
from src.feature.preprocessor import ImagePreprocessor
from src.retrieval.indexer import FAISSIndexer
from src.data.manager import MetadataManager
from src.utils.logger import setup_logger

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logger
logger = setup_logger('run_retrieval')


def load_config(config_path: str = None) -> dict:
    """Load configuration file"""
    if config_path is None:
        config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_descriptors(descriptor_path: Path) -> tuple:
    """Load pre-computed descriptors"""
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Descriptor file not found: {descriptor_path}")
    
    data = torch.load(descriptor_path, map_location='cpu')
    
    if 'descriptors' in data:
        descriptors = data['descriptors']
        if isinstance(descriptors, torch.Tensor):
            descriptors = descriptors.numpy()
    else:
        # Handle old format
        descriptors = data.numpy() if isinstance(data, torch.Tensor) else data
    
    image_ids = data.get('image_ids', [])
    
    return descriptors, image_ids


def extract_query_descriptor(
    query_path: Path,
    extractor: FeatureExtractor,
    aggregator: FeatureAggregator,
    preprocessor: ImagePreprocessor
) -> np.ndarray:
    """Extract descriptor for a single query image"""
    
    # Preprocess
    img_tensor = preprocessor.process(query_path)
    img_batch = img_tensor.unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = extractor.extract(img_batch)
        
        # Aggregate
        if aggregator.agg_type == "vlad":
            descriptor = aggregator.aggregate(features[0])
        else:
            descriptor = aggregator.aggregate(features)
            descriptor = descriptor[0] if descriptor.dim() > 1 else descriptor
        
        descriptor = descriptor.cpu().numpy()
    
    return descriptor


def perform_retrieval(
    query_descriptor: np.ndarray,
    database_descriptors: np.ndarray,
    indexer: FAISSIndexer,
    top_k: int = 10
) -> tuple:
    """Perform similarity search"""
    
    # Search
    similarities, indices = indexer.search(query_descriptor.reshape(1, -1), top_k)
    
    return similarities[0], indices[0]


def main():
    parser = argparse.ArgumentParser(description="Run image retrieval")
    
    # Input options
    parser.add_argument('--query', type=str, default=None,
                       help='Query image path or directory')
    parser.add_argument('--database_desc', type=str, default=None,
                       help='Pre-computed database descriptors')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Metadata CSV file')
    
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
    
    # Retrieval options
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top results to return')
    parser.add_argument('--batch', action='store_true',
                       help='Process all queries in directory')
    
    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--profile', type=str, default=None,
                       help='Use predefined profile from config')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
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
    
    # Get model and aggregator settings
    model_type = config['model']['extractor']['type']
    agg_type = config['model']['aggregator']['type']
    
    logger.info("="*60)
    logger.info("Retrieval Configuration")
    logger.info("="*60)
    logger.info(f"Model: {model_type}")
    logger.info(f"Aggregator: {agg_type}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info("="*60)
    
    # Load database descriptors
    if args.database_desc:
        db_desc_path = Path(args.database_desc)
    else:
        # Auto-detect based on model and aggregator
        desc_dir = Path(config['data']['descriptor_dir'])
        model_suffix = model_type.replace('/', '_')
        agg_suffix = agg_type
        db_desc_path = desc_dir / f"database_descriptors_{model_suffix}_{agg_suffix}.pt"
    
    logger.info(f"Loading database descriptors from {db_desc_path}")
    db_descriptors, db_image_ids = load_descriptors(db_desc_path)
    logger.info(f"  Loaded {len(db_descriptors)} descriptors")
    logger.info(f"  Dimension: {db_descriptors.shape[1]}")
    
    # Load metadata
    metadata_manager = None
    if args.metadata or config['data'].get('database_metadata'):
        metadata_path = args.metadata or config['data']['database_metadata']
        if Path(metadata_path).exists():
            logger.info(f"Loading metadata from {metadata_path}")
            metadata_manager = MetadataManager(metadata_path)
    
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
        target_size=None,
        mode='query',
        patch_size=extractor.get_patch_size()
    )
    
    # Initialize indexer
    logger.info("Building search index...")
    indexer = FAISSIndexer(use_gpu=(args.device == 'cuda'))
    indexer.build_index(db_descriptors)
    
    # Process queries
    query_path = Path(args.query or config['data']['query_dir'])
    
    all_results = []
    
    if query_path.is_file():
        # Single query
        logger.info(f"Processing query: {query_path}")
        
        # Time tracking
        start_time = time.time()
        times = {}
        
        # Preprocessing
        t0 = time.time()
        query_desc = extract_query_descriptor(
            query_path, extractor, aggregator, preprocessor
        )
        times['feature_extraction'] = (time.time() - t0) * 1000
        
        # Search
        t0 = time.time()
        similarities, indices = perform_retrieval(
            query_desc, db_descriptors, indexer, args.top_k
        )
        times['search'] = (time.time() - t0) * 1000
        
        times['total'] = (time.time() - start_time) * 1000
        
        # Display results
        logger.info("\nRetrieval Results:")
        logger.info("-" * 40)
        
        results = []
        for rank, (sim, idx) in enumerate(zip(similarities, indices), 1):
            result = {
                'rank': rank,
                'score': float(sim),
                'image_id': db_image_ids[idx] if idx < len(db_image_ids) else str(idx),
                'index': int(idx)
            }
            
            # Add metadata if available
            if metadata_manager:
                metadata = metadata_manager.get_metadata(db_image_ids[idx])
                if metadata:
                    result.update(metadata)
            
            results.append(result)
            
            if args.verbose or rank <= 5:
                logger.info(f"Rank {rank}: {result['image_id']} (score: {sim:.4f})")
                if metadata_manager and 'center_lat' in result:
                    logger.info(f"  Location: ({result['center_lat']:.6f}, {result['center_long']:.6f})")
        
        all_results.append({
            'query': str(query_path),
            'results': results,
            'times': times
        })
        
        # Performance analysis
        if args.verbose:
            logger.info("\n" + "="*60)
            logger.info("Performance Analysis:")
            logger.info("="*60)
            logger.info(f"Feature extraction  : {times['feature_extraction']:7.2f} ms")
            logger.info(f"Search              : {times['search']:7.2f} ms")
            logger.info(f"Total               : {times['total']:7.2f} ms")
        
    elif query_path.is_dir():
        # Batch processing
        query_images = list(query_path.glob('*.jpg')) + list(query_path.glob('*.png'))
        logger.info(f"Processing {len(query_images)} queries from {query_path}")
        
        for query_img in query_images:
            logger.info(f"Processing: {query_img.name}")
            
            # Extract descriptor
            query_desc = extract_query_descriptor(
                query_img, extractor, aggregator, preprocessor
            )
            
            # Search
            similarities, indices = perform_retrieval(
                query_desc, db_descriptors, indexer, args.top_k
            )
            
            # Store results
            results = []
            for rank, (sim, idx) in enumerate(zip(similarities, indices), 1):
                result = {
                    'rank': rank,
                    'score': float(sim),
                    'image_id': db_image_ids[idx] if idx < len(db_image_ids) else str(idx),
                    'index': int(idx)
                }
                
                if metadata_manager:
                    metadata = metadata_manager.get_metadata(db_image_ids[idx])
                    if metadata:
                        result.update(metadata)
                
                results.append(result)
            
            all_results.append({
                'query': str(query_img),
                'results': results
            })
            
            if not args.verbose:
                # Show top match only
                logger.info(f"  Top match: {results[0]['image_id']} (score: {results[0]['score']:.4f})")
    
    # Save results
    if args.output or len(all_results) > 1:
        output_path = args.output or Path(config['data']['results_dir']) / 'retrieval_results.json'
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'config': {
                    'model': model_type,
                    'aggregator': agg_type,
                    'top_k': args.top_k
                },
                'queries': all_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    logger.info("\nRetrieval complete!")


if __name__ == "__main__":
    main()