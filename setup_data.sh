#!/bin/bash

mkdir -p data
cd data

unzip train_data.csv.zip
unzip test_sequences.csv.zip

cd ..

python3 utils/csv_to_parquet.py --file data/train_data.csv --output_file data/train_data.parquet