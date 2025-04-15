import os
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Combine ensemble and correction predictions")
    parser.add_argument("--ensemble_pred_path",
                        required=True,
                        type=str,
                        help="Path to ensemble prediction parquet file")
    parser.add_argument("--ensemble_pred_count",
                        required=True,
                        type=int,
                        help="Number of models in the ensemble")
    parser.add_argument("--correction_pred_path", 
                        required=True,
                        type=str,
                        help="Path to correction model prediction parquet file")
    parser.add_argument("--out_dir", 
                        required=True,
                        type=str,
                        help="Output directory for the final prediction")
    
    return parser.parse_args()


def load_predictions(ensemble_path, correction_path):
    """Load ensemble and correction prediction dataframes.
    
    Args:
        ensemble_path (Path): Path to ensemble predictions parquet file
        correction_path (Path): Path to correction predictions parquet file
        
    Returns:
        tuple: Tuple containing (ensemble_df, correction_df)
    """
    df_ensemble = pd.read_parquet(ensemble_path)
    df_correction = pd.read_parquet(correction_path)
    
    return df_ensemble, df_correction


def combine_predictions(df_ensemble, df_correction, ensemble_count):
    """Combine ensemble and correction predictions with weighted average.
    
    Args:
        df_ensemble (DataFrame): Ensemble predictions dataframe
        df_correction (DataFrame): Correction predictions dataframe
        ensemble_count (int): Number of models in the ensemble
        
    Returns:
        DataFrame: Combined predictions dataframe
    """
    # Calculate weights
    weight_ensemble = ensemble_count / (ensemble_count + 1)
    weight_correction = 1 / (ensemble_count + 1)
    
    # Apply weighted average to reactivity values
    df_ensemble["reactivity_2A3_MaP"] = (
        weight_ensemble * df_ensemble["reactivity_2A3_MaP"] + 
        weight_correction * df_correction["reactivity_2A3_MaP"]
    )
    
    return df_ensemble


def save_predictions(df_combined, output_dir):
    """Save combined predictions to output directory.
    
    Args:
        df_combined (DataFrame): Combined predictions dataframe
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir) / "final_pred.parquet"
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df_combined.to_parquet(output_path, index=False)
    print(f"Final predictions saved to: {output_path}")


def main():
    """Main function to execute the prediction combination pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load predictions
    ensemble_path = Path(args.ensemble_pred_path)
    correction_path = Path(args.correction_pred_path)
    df_ensemble, df_correction = load_predictions(ensemble_path, correction_path)
    
    # Combine predictions
    df_combined = combine_predictions(
        df_ensemble, 
        df_correction, 
        args.ensemble_pred_count
    )
    
    # Save final predictions
    save_predictions(df_combined, args.out_dir)


if __name__ == "__main__":
    main()