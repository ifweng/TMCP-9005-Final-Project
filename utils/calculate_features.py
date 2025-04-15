import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np 
import pandas as pd
import concurrent.futures
import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNA structure prediction pipeline")
    parser.add_argument("--train_path", required=True, type=str, 
                        help="Path to training data in parquet format")
    parser.add_argument("--test_path", required=True, type=str,
                        help="Path to test data in CSV format")
    parser.add_argument("--arnie_config_path", required=True, type=str,
                        help="Path to ARNIE configuration file")
    parser.add_argument("--arnie_path", required=True, type=str,
                        help="Path to ARNIE installation directory")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Directory to save output data")
    parser.add_argument("--nprocs", default=50, type=int,
                        help="Number of concurrent processes to use")
    return parser.parse_args()

def setup_arnie(arnie_config_path, arnie_path):
    """Setup ARNIE environment and import necessary modules."""
    # Resolve paths and set environment variables
    arnie_config_path = str(Path(arnie_config_path).resolve())
    os.environ['ARNIEFILE'] = arnie_config_path
    
    # Add ARNIE to sys.path
    arnie_path = str(Path(arnie_path).resolve())
    sys.path.append(arnie_path)
    
    # Import ARNIE modules
    from arnie.bpps import bpps
    from arnie.pk_predictors import pk_predict
    from arnie.mfe import mfe
    
    return bpps, pk_predict, mfe

def calc_arnie(seq, seqid, aim_dir, bpps_func):
    """Calculate base pair probabilities using ARNIE and save to file."""
    res = bpps_func(seq, package="eternafold")
    outpath = aim_dir / f"{seqid}.npy"
    np.save(outpath, res)
    return seqid

def process_bppm(data, aim_dir, nprocs, bpps_func):
    """Process base pair probability matrices for a dataset."""
    aim_dir.mkdir(parents=True, exist_ok=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = {}
        for ind, seqid, seq in data[['sequence_id', 'sequence']].itertuples():
            ft = executor.submit(calc_arnie, seq=seq, seqid=seqid, aim_dir=aim_dir, bpps_func=bpps_func)
            futures[ft] = seqid

        for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                ft.result()
            except Exception as exc:
                seqid = futures[ft]
                print(f"Error occurred while processing {seqid}: {exc}")

def calc_mfe(seq, mfe_func):
    """Calculate minimum free energy structure prediction."""
    return mfe_func(seq, package="eternafold")

def calc_ipknot(seq, pk_predict_func):
    """Calculate pseudoknot structure prediction using IPknot."""
    return pk_predict_func(seq, 'ipknot', refinement=1, cpu=1)

def process_structure_predictions(data, aim_dir, nprocs, calc_func, filename):
    """Process structure predictions for a dataset and save results to a JSON file."""
    aim_dir.mkdir(parents=True, exist_ok=True)
    
    seqid2seq = {seqid: seq for ind, seqid, seq in data[['sequence_id', 'sequence']].itertuples()}
    seq2structure = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = {}
        for seqid, seq in tqdm.tqdm(seqid2seq.items(), total=len(seqid2seq)):
            ft = executor.submit(calc_func, seq=seq)
            futures[ft] = seqid

        for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            seqid = futures[ft]
            try:
                result = ft.result()
                seq2structure[seqid] = result
            except Exception as exc:
                print(f"Error occurred while processing {seqid}: {exc}")
    
    with open(aim_dir / filename, "w") as out:
        json.dump(seq2structure, out)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup ARNIE
    bpps, pk_predict, mfe = setup_arnie(args.arnie_config_path, args.arnie_path)
    
    # Load datasets
    print("Loading datasets...")
    train_data = pd.read_parquet(args.train_path)
    test_data = pd.read_csv(args.test_path)
    
    # Create base data directory
    data_dir = Path(args.data_dir)
    
    # Calculate BPPM for train data
    print("Calculating: TRAIN ETERNAFOLD BPPM")
    train_bpp_dir = data_dir / "BPP" / "eternafold" / "train"
    process_bppm(train_data, train_bpp_dir, args.nprocs, bpps)
    
    # Calculate BPPM for test data
    print("Calculating: TEST ETERNAFOLD BPPM")
    test_bpp_dir = data_dir / "BPP" / "eternafold" / "test"
    process_bppm(test_data, test_bpp_dir, args.nprocs, bpps)
    
    # Calculate MFE structures for train data
    print("Calculating: TRAIN ETERNA MFE BRACKETS")
    train_brackets_dir = data_dir / "brackets_train"
    process_structure_predictions(
        train_data, 
        train_brackets_dir, 
        args.nprocs,
        lambda seq: calc_mfe(seq, mfe),
        "eterna.json"
    )
    
    # Calculate MFE structures for test data
    print("Calculating: TEST ETERNA MFE BRACKETS")
    test_brackets_dir = data_dir / "brackets_test"
    process_structure_predictions(
        test_data, 
        test_brackets_dir, 
        args.nprocs,
        lambda seq: calc_mfe(seq, mfe),
        "eterna.json"
    )
    
    # Calculate IPknot structures for train data
    print("Calculating: TRAIN IPKNOT MFE BRACKETS")
    process_structure_predictions(
        train_data, 
        train_brackets_dir, 
        args.nprocs,
        lambda seq: calc_ipknot(seq, pk_predict),
        "ipknot.json"
    )
    
    # Calculate IPknot structures for test data (excluding last 20 sequences)
    print("Calculating: TEST IPKNOT MFE BRACKETS")
    process_structure_predictions(
        test_data.iloc[:-20], 
        test_brackets_dir, 
        args.nprocs,
        lambda seq: calc_ipknot(seq, pk_predict),
        "ipknot.json"
    )

if __name__ == "__main__":
    main()