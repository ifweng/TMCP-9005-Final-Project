#!/usr/bin/env python3
"""
This script searches for model weights in child directories of a given source directory 
and copies them to a specified output directory.
"""

import os
import shutil
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy model weight files from nested directories to a consolidated location"
    )
    parser.add_argument(
        "--source",
        required=True,
        type=str,
        help="Source directory containing model directories"
    )
    parser.add_argument(
        "--out_dir", 
        required=True,
        type=str,
        help="Output directory where model weights will be copied"
    )
    
    return parser.parse_args()


def get_subdirectories(directory):
    """Get a list of subdirectory paths from a directory.
    
    Args:
        directory (Path): Path to the parent directory
        
    Returns:
        list: List of Path objects for subdirectories
    """
    return [directory / item for item in os.listdir(directory) 
            if (directory / item).is_dir()]


def copy_model_weights(source_dir, output_dir):
    """Copy model weight files from source directory structure to output directory.
    
    Args:
        source_dir (Path): Source directory containing model directories
        output_dir (Path): Output directory where model weights will be copied
    """
    # Get experiment directories (first level subdirectories)
    experiment_dirs = get_subdirectories(source_dir)
    
    total_models_copied = 0
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"Processing experiment: {exp_name}")
        
        # Get model directories (second level subdirectories)
        model_dirs = get_subdirectories(exp_dir)
        
        # Create output directory for this experiment
        exp_output_dir = output_dir / exp_name
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Copy model weights
        for model_num, model_dir in enumerate(model_dirs):
            source_path = model_dir / 'fst_model' / 'pos_aug_sgd' / 'model.pth'
            destination_path = exp_output_dir / f'model_{model_num}.pth'
            
            if source_path.exists():
                shutil.copy(source_path, destination_path)
                print(f"  Copied: {source_path} -> {destination_path}")
                total_models_copied += 1
            else:
                print(f"  Warning: Model file not found at {source_path}")
    
    print(f"\nProcess completed. Total models copied: {total_models_copied}")


def main():
    """Main function to execute model weights copying process."""
    args = parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.out_dir)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Searching for model weights in: {source_dir}")
    print(f"Output directory for copied weights: {output_dir}\n")
    
    copy_model_weights(source_dir, output_dir)


if __name__ == "__main__":
    main()