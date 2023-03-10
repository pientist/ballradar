python train.py \
--trial 810 \
--model player_ball \
--macro_type player_poss \
--target_type transition \
--macro_weight 1 \
--bidirectional \
--flip_pitch \
--n_players 11 \
--n_features 6 \
--macro_embed_dim 16 \
--macro_rnn_dim 128 \
--micro_embed_dim 128 \
--micro_rnn_dim 256 \
--dropout 0.2 \
--n_layers 2 \
--n_epochs 50 \
--start_lr 5e-4 \
--min_lr 1e-5 \
--batch_size 448 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda