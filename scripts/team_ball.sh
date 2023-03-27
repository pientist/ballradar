#!/bin/bash

python train.py \
--trial 202 \
--model team_ball \
--macro_type team_poss \
--target_type gk \
--macro_weight 20 \
--reality_weight 1 \
--bidirectional \
--flip_pitch \
--n_players 10 \
--n_features 6 \
--z_dim 128 \
--macro_rnn_dim 128 \
--micro_rnn_dim 256 \
--n_layers 2 \
--dropout 0.2 \
--n_epochs 50 \
--start_lr 5e-4 \
--min_lr 1e-5 \
--batch_size 896 \
--print_every_batch 20 \
--save_every_epoch 50 \
--seed 100 \
--cuda