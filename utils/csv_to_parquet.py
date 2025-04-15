import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert CSV file to Parquet format for faster data loading")
    parser.add_argument("--file",
                        required=True,
                        type=str,
                        help="Input CSV file path")
    parser.add_argument("--output_file", 
                        required=True,
                        type=str,
                        help="Output Parquet file path")
    
    return parser.parse_args()


def convert_csv_to_parquet(input_file, output_file):
    """Convert a CSV file to Parquet format.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output Parquet file
    """
    try:
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting to Parquet format...")
        df.to_parquet(output_file)
        print(f"Successfully saved Parquet file to: {output_file}")
        print(f"Converted {len(df)} rows of data")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise


def main():
    """Main function to execute the CSV to Parquet conversion."""
    args = parse_args()
    convert_csv_to_parquet(args.file, args.output_file)


if __name__ == "__main__":
    main()