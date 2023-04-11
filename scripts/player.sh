#!/bin/bash

python train.py \
--trial 502 \
--model pe_lstm \
--target_type player_poss \
--bidirectional \
--flip_pitch \
--n_players 11 \
--n_features 6 \
--embed_dim 4 \
--rnn_dim 10 \
--n_layers 2 \
--dropout 0.3 \
--n_epochs 50 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 896 \
--print_every_batch 10 \
--save_every_epoch 50 \
--seed 100 \
--cuda