python train.py \
--trial 110 \
--model player_ball \
--macro_type player_poss \
--target_type ball \
--macro_weight 20 \
--rloss_weight 1 \
--masking 0.8 \
--bidirectional \
--train_fito \
--valid_fito \
--train_metrica \
--valid_metrica \
--flip_pitch \
--n_players 11 \
--n_features 6 \
--macro_ppe \
--macro_fpe \
--macro_fpi \
--macro_pe_dim 16 \
--macro_pi_dim 16 \
--macro_rnn_dim 256 \
--micro_pi_dim 128 \
--micro_rnn_dim 256 \
--n_epochs 50 \
--start_lr 0.0005 \
--min_lr 1e-6 \
--batch_size 224 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda