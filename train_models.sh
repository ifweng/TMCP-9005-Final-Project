#!/bin/bash

#this script runs training procedure for each model in ensemble

bpp_path=data/BPP/eternafold/train
train_path=data/train_data.parquet
save_dir=train_results
bracket_path=data/brackets_train

epochs=270
device=0
num_workers=20
wd=0.05
lr_max=5e-3
pct_start=0.05
batch_cnt=1791
sgd_lr=5e-5
sgd_epochs=25
sgd_batch_cnt=500
sgd_wd=0.05
nfolds=1000


mkdir -p $save_dir/brks_lengths
mkdir -p $save_dir/dms2a3
mkdir -p $save_dir/lengths
mkdir -p $save_dir/se_thousands
mkdir -p $save_dir/weights_thousands

weights_thousands_params=configs/folds_and_seeds_no_se.txt
se_thousands_params=configs/folds_and_seeds_se.txt


#train models without se-blocks and without brackets
cat $weights_thousands_params | while read line 
do
    IFS=',' read item1 item2 <<< "${line}"

    fold=$item1
    seed=$item2
    python3 utils/train_uni_adjnet.py --bpp_path $bpp_path  --train_path $train_path --out_path  $save_dir/weights_thousands/model_fold${fold}_seed${seed} --device $device --num_workers $num_workers --wd $wd --epoch $epochs --lr_max $lr_max --pct_start $pct_start --batch_cnt $batch_cnt --sgd_lr $sgd_lr --sgd_epochs $sgd_epochs --sgd_batch_cnt $sgd_batch_cnt --sgd_wd $sgd_wd --fold $fold --nfolds $nfolds --pos_embedding dyn --adj_ks 3 --seed $seed
done


# train models with se-blocks and without brackets
cat $se_thousands_params | while read line 
do
    IFS=',' read item1 item2 <<< "${line}"

    fold=$item1
    seed=$item2
    python3 utils/train_uni_adjnet.py --bpp_path $bpp_path  --train_path $train_path --out_path  $save_dir/se_thousands/model_fold${fold}_seed${seed} --device $device --num_workers $num_workers --wd $wd --epoch $epochs --lr_max $lr_max --pct_start $pct_start --batch_cnt $batch_cnt --sgd_lr $sgd_lr --sgd_epochs $sgd_epochs --sgd_batch_cnt $sgd_batch_cnt --sgd_wd $sgd_wd --fold $fold --nfolds $nfolds --pos_embedding dyn --adj_ks 3 --seed $seed --use_se
done

# train model without se-blocks and without brackets on split by sequence length
fold=0
seed=2023
python3 utils/train_uni_adjnet.py --bpp_path $bpp_path  --train_path $train_path --out_path  $save_dir/lengths/model_fold${fold}_seed${seed} --device $device --num_workers $num_workers --wd $wd --epoch $epochs --lr_max $lr_max --pct_start $pct_start --batch_cnt $batch_cnt --sgd_lr $sgd_lr --sgd_epochs $sgd_epochs --sgd_batch_cnt $sgd_batch_cnt --sgd_wd $sgd_wd --fold $fold --nfolds $nfolds --pos_embedding dyn --adj_ks 3 --seed $seed --split length


# train model without se-blocks and with brackets on split by sequence length
fold=0
seed=2023
python3 utils/train_uni_adjnet.py --bpp_path $bpp_path  --train_path $train_path --out_path  $save_dir/brks_lengths/model_fold${fold}_seed${seed} --device $device --num_workers $num_workers --wd $wd --epoch $epochs --lr_max $lr_max --pct_start $pct_start --batch_cnt $batch_cnt --sgd_lr $sgd_lr --sgd_epochs $sgd_epochs --sgd_batch_cnt $sgd_batch_cnt --sgd_wd $sgd_wd --fold $fold --nfolds $nfolds --pos_embedding dyn --adj_ks 3 --seed $seed --split length --brackets $bracket_path/eterna.json $bracket_path/ipknot.json


# train model without se-blocks and without brackets on split by sequence length
# this model additionally takes dms reactivities as input and outputs 2a3 reactivities
fold=0
seed=2023
python3 utils/train_uni_adjnet_dms22a3.py --bpp_path $bpp_path  --train_path $train_path --out_path  $save_dir/dms2a3/model_fold${fold}_seed${seed} --device $device --num_workers $num_workers --wd $wd --epoch $epochs --lr_max $lr_max --pct_start $pct_start --batch_cnt $batch_cnt --sgd_lr $sgd_lr --sgd_epochs $sgd_epochs --sgd_batch_cnt $sgd_batch_cnt --sgd_wd $sgd_wd --fold $fold --nfolds $nfolds --pos_embedding dyn --adj_ks 3 --seed $seed --split length --pred_mode dms_2a3


mkdir -p model_weights
python3 utils/gather_model_weights.py --source $save_dir --out_dir model_weights