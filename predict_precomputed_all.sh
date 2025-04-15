#!/bin/bash

bpp_path=data/BPP/eternafold/test
test_path=data/test_sequences.csv
model_weights_dir=precomputed/model_weights
out_dir=predictions/single
mkdir -p predictions/single
mkdir -p predictions/main
mkdir -p predictions/final
bracket_path=data/brackets_test

# models trained on kfold 1000 split without se-blocks and brackets
# total number - 10 models
for i in {0..9}
do
   python3 utils/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/weights_thousands/model_$i.pth --out_path $out_dir --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20
done

# models trained on kfold 1000 split with se-blocks and without brackets
# total number - 15 models
for i in {0..14}
do
   python3 utils/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/se_thousands/model_$i.pth --out_path $out_dir --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20 --use_se
done

# model trained on split by length without se-blocks and without brackets
python3 utils/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/lengths/model_0.pth --out_path $out_dir --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20

# model trained on split by length without se-blocks and with brackets
python3 utils/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/brks_lengths/model_0.pth --out_path $out_dir --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20 --brackets $bracket_path/ipknot.json $bracket_path/eterna.json

# calculates main prediction by averaging individual predictions made by each model
python3 utils/calculate_main_pred.py --pred_dir $out_dir --out_dir predictions/main

#predicts and adds correction predicted by dms-to-2a3 model
python3 utils/predict_dms22a3.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/dms2a3/model_0.pth --out_path predictions/main --react_preds_path predictions/main/ensemble_pred.parquet  --device 0 --pos_embedding dyn --adj_ks 3 --pred_mode dms_2a3

python3 utils/calculate_final_pred.py --ensemble_pred_path predictions/main/ensemble_pred.parquet --ensemble_pred_count 27 --correction_pred_path predictions/main/submit_dms2a3_model_0.parquet --out_dir predictions/final








