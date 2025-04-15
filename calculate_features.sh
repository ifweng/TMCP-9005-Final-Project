#!/bin/bash

python3 utils/calculate_features.py --train_path data/train_data.parquet --test_path data/test_sequences.csv --arnie_path tools/arnie/src --arnie_config_path tools/arnie/arnie_config.txt --data_dir data