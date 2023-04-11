python train.py \
--trial 302 \
--model pi_vrnn \
--target_type ball \
--train_fito \
--valid_fito \
--train_metrica \
--valid_metrica \
--flip_pitch \
--n_players 11 \
--n_features 6 \
--context_dim 128 \
--rnn_h_dim 256 \
--n_layers 2 \
--dropout 0.2 \
--vae_h_dim 128 \
--vae_z_dim 16 \
--n_epochs 50 \
--start_lr 5e-4 \
--min_lr 1e-5 \
--batch_size 1792 \
--print_every_batch 20 \
--save_every_epoch 50 \
--seed 100 \
--cuda