import os
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Average predictions from multiple models")
    parser.add_argument("--pred_dir",
                        required=True,
                        type=str,
                        help="Directory containing prediction parquet files")
    parser.add_argument("--out_dir", 
                        required=True,
                        type=str,
                        help="Output directory for the ensemble prediction")
    
    return parser.parse_args()


def get_prediction_files(pred_dir):
    """Get list of prediction files from the prediction directory.
    
    Args:
        pred_dir (str): Path to directory containing prediction files
        
    Returns:
        list: List of prediction file paths
    """
    pred_path = Path(pred_dir)
    pred_files = os.listdir(pred_dir)
    return [pred_path / file for file in pred_files]


def average_predictions(file_paths):
    """Average predictions from multiple parquet files.
    
    Args:
        file_paths (list): List of paths to prediction parquet files
        
    Returns:
        DataFrame: Averaged predictions dataframe
    """
    # Read the first file to initialize
    df_ensemble = pd.read_parquet(file_paths[0])
    
    # Add predictions from remaining files
    for file_path in file_paths[1:]:
        temp_df = pd.read_parquet(file_path)
        df_ensemble["reactivity_DMS_MaP"] += temp_df["reactivity_DMS_MaP"]
        df_ensemble["reactivity_2A3_MaP"] += temp_df["reactivity_2A3_MaP"]
    
    # Calculate averages
    num_files = len(file_paths)
    df_ensemble["reactivity_DMS_MaP"] /= num_files
    df_ensemble["reactivity_2A3_MaP"] /= num_files
    
    return df_ensemble


def save_predictions(df_ensemble, output_dir):
    """Save ensemble predictions to output directory.
    
    Args:
        df_ensemble (DataFrame): Ensemble predictions dataframe
        output_dir (str): Output directory path
    """
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    output_file = out_path / "ensemble_pred.parquet"
    df_ensemble.to_parquet(output_file, index=False)
    print(f"Ensemble predictions saved to: {output_file}")


def main():
    """Main function to execute the ensemble prediction pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Get prediction files
    pred_files = get_prediction_files(args.pred_dir)
    
    if not pred_files:
        print(f"No prediction files found in {args.pred_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files to ensemble")
    
    # Average predictions
    df_ensemble = average_predictions(pred_files)
    
    # Save ensemble predictions
    save_predictions(df_ensemble, args.out_dir)


if __name__ == "__main__":
    main()