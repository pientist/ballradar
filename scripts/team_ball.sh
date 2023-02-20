#!/bin/bash

python train.py \
--trial 707 \
--model team_ball \
--macro_type team_poss \
--target_type ball \
--macro_weight 50 \
--reality_weight 1 \
--prev_out_aware \
--bidirectional \
--flip_pitch \
--n_features 6 \
--embed_dim 128 \
--macro_rnn_dim 32 \
--micro_rnn_dim 256 \
--n_layers 2 \
--n_epochs 50 \
--start_lr 5e-4 \
--min_lr 1e-5 \
--batch_size 1792 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda