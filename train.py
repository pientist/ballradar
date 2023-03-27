import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SoccerDataset
from models import load_model
from models.utils import (
    calc_class_acc,
    calc_real_loss,
    calc_speed,
    calc_trace_dist,
    get_params_str,
    num_trainable_params,
)

# Modified from https://github.com/ezhan94/multiagent-programmatic-supervision/blob/master/train.py


# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def loss_str(losses: dict):
    ret = ""
    for key, value in losses.items():
        ret += " {}: {:.4f} |".format(key, np.mean(value))
    # if len(losses) > 1:
    #     ret += " total_loss: {:.4f} |".format(sum(losses.values()))
    return ret[:-2]


def hyperparams_str(epoch, hp):
    ret = "\nEpoch {:d}".format(epoch)
    if hp["pretrain"]:
        ret += " (pretrain)"
    return ret


# For one epoch
def run_epoch(model: nn.DataParallel, optimizer: torch.optim.Adam, train=False, print_every=50):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()

    loader = train_loader if train else test_loader
    n_batches = len(loader)

    if model.module.model_type == "classifier":
        loss_dict = {"ce_loss": [], "accuracy": []}

    elif model.module.model_type == "regressor":
        if "rloss_weight" in model.module.params:
            loss_dict = {"mse_loss": [], "real_loss": [], "pos_error": []}
        else:
            loss_dict = {"mse_loss": [], "pos_error": []}

    elif model.module.model_type == "macro_classifier":
        loss_dict = {"ce_loss": [], "micro_ce_loss": [], "macro_acc": [], "micro_acc": []}

    elif model.module.model_type == "macro_regressor":
        if "rloss_weight" in model.module.params and model.module.params["rloss_weight"] > 0:
            loss_dict = {"ce_loss": [], "mse_loss": [], "real_loss": [], "macro_acc": [], "micro_pos_error": []}
        else:
            loss_dict = {"ce_loss": [], "mse_loss": [], "macro_acc": [], "micro_pos_error": []}

    for batch_idx, data in enumerate(loader):
        if model.module.model_type == "classifier":
            input = data[0].to(default_device)
            target = data[1].to(default_device)

            if train:
                out = model(input).transpose(1, 2)
            else:
                with torch.no_grad():
                    out = model(input).transpose(1, 2)

            loss = nn.CrossEntropyLoss()(out, target)
            loss_dict["ce_loss"] += [loss.item()]
            loss_dict["accuracy"] += [calc_class_acc(out, target)]

        elif model.module.model_type == "regressor":
            input = data[0].to(default_device)
            target = data[1].to(default_device)

            if train:
                out = model(input)
            else:
                with torch.no_grad():
                    out = model(input)

            if "speed_loss" in model.module.params and model.module.params["speed_loss"]:
                out = calc_speed(out)

            loss = nn.MSELoss()(out, target)
            loss_dict["mse_loss"] += [loss.item()]

            if "rloss_weight" in model.module.params:
                rloss_weight = model.module.params["rloss_weight"]
                if rloss_weight > 0:
                    n_features = model.module.params["n_features"]
                    real_loss = calc_real_loss(out[:, :, 0:2], input, n_features) * rloss_weight
                    loss_dict["real_loss"] += [real_loss.item()]
                    loss += real_loss

            if model.module.target_type == "gk":
                team1_pos_error = calc_trace_dist(out[:, :, 0:2], target[:, :, 0:2])
                team2_pos_error = calc_trace_dist(out[:, :, 2:4], target[:, :, 2:4])
                loss_dict["pos_error"] += [(team1_pos_error + team2_pos_error) / 2]
            else:
                loss_dict["pos_error"] += [calc_trace_dist(out[:, :, 0:2], target[:, :, 0:2])]

        elif model.module.model_type.startswith("macro"):
            input = data[0].to(default_device)
            macro_target = data[1].to(default_device)
            micro_target = data[2].to(default_device)

            if train:
                out = model(input, macro_target, micro_target)
            else:
                with torch.no_grad():
                    out = model(input, macro_target, micro_target)

            micro_out_dim = model.module.micro_out_dim  # 4 if target_type == "gk" else 2
            macro_out = out[:, :, :-micro_out_dim].transpose(1, 2)
            macro_weight = model.module.params["macro_weight"]
            macro_loss = nn.CrossEntropyLoss()(macro_out, macro_target) * macro_weight

            loss_dict["ce_loss"] += [macro_loss.item()]
            loss_dict["macro_acc"] += [calc_class_acc(macro_out, macro_target)]

            if model.module.model_type == "macro_classifier":
                micro_out = out[:, :, -micro_out_dim:].transpose(1, 2)
                micro_loss = nn.CrossEntropyLoss()(micro_out, micro_target)
                loss = macro_loss + micro_loss

                loss_dict["micro_ce_loss"] += [micro_loss.item()]
                loss_dict["micro_acc"] += [calc_class_acc(micro_out, micro_target)]

            else:  # model.module.model_type == "macro_regressor"
                micro_out = out[:, :, -micro_out_dim:]
                if "speed_loss" in model.module.params and model.module.params["speed_loss"]:
                    micro_out = calc_speed(micro_out)

                micro_loss = nn.MSELoss()(micro_out, micro_target)
                loss = macro_loss + micro_loss
                if "rloss_weight" in model.module.params:
                    rloss_weight = model.module.params["rloss_weight"]
                    if rloss_weight > 0:
                        n_features = model.module.params["n_features"]
                        real_loss = calc_real_loss(micro_out[:, :, 0:2], input, n_features) * rloss_weight
                        loss_dict["real_loss"] += [real_loss.item()]
                        loss += real_loss

                loss_dict["mse_loss"] += [micro_loss.item()]
                if model.module.target_type == "gk":
                    team1_pos_error = calc_trace_dist(micro_out[:, :, 0:2], micro_target[:, :, 0:2])
                    team2_pos_error = calc_trace_dist(micro_out[:, :, 2:4], micro_target[:, :, 2:4])
                    loss_dict["micro_pos_error"] += [(team1_pos_error + team2_pos_error) / 2]
                else:
                    loss_dict["micro_pos_error"] += [calc_trace_dist(micro_out[:, :, 0:2], micro_target[:, :, 0:2])]

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), clip)
            optimizer.step()

        if train and batch_idx % print_every == 0:
            print(f"[{batch_idx:>{len(str(n_batches))}d}/{n_batches}]  {loss_str(loss_dict)}")

    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)  # /= len(loader.dataset)

    return loss_dict


# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="player_ball")
parser.add_argument("--macro_type", type=str, required=False, default="team_poss", help="type of macro-intents")
parser.add_argument("--target_type", type=str, required=False, default="ball", help="gk, ball, or team_poss")
parser.add_argument("--macro_weight", type=float, required=False, default=50, help="weight for the macro-intent loss")
parser.add_argument("--rloss_weight", type=float, required=False, default=0, help="weight for the reality loss")
parser.add_argument("--speed_loss", action="store_true", default=False, help="include speed loss in MSE")
parser.add_argument("--masking", type=float, required=False, default=1, help="masking proportion of the target")
parser.add_argument("--prev_out_aware", action="store_true", default=False, help="make RNN refer to previous outputs")
parser.add_argument("--bidirectional", action="store_true", default=False, help="make RNN bidirectional")

parser.add_argument("--train_fito", action="store_true", default=False, help="Use Fitogether data for training")
parser.add_argument("--valid_fito", action="store_true", default=False, help="Use Fitogether data for validation")
parser.add_argument("--train_metrica", action="store_true", default=False, help="Use Metrica data for training")
parser.add_argument("--valid_metrica", action="store_true", default=False, help="Use Metrica data for validation")
parser.add_argument("--flip_pitch", action="store_true", default=False, help="augment data by flipping the pitch")
parser.add_argument("--n_features", type=int, required=False, default=2, help="num features")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="starting learning rate")
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=int, required=False, default=10, help="gradient clipping")
parser.add_argument("--print_every_batch", type=int, required=False, default=50, help="periodically print performance")
parser.add_argument("--save_every_epoch", type=int, required=False, default=10, help="periodically save model")
parser.add_argument("--pretrain_time", type=int, required=False, default=0, help="num epochs to train macro policy")
parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")
parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
parser.add_argument("--cont", action="store_true", default=False, help="continue training previous best model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best test loss")

args, _ = parser.parse_known_args()


if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    default_device = "cuda:0"

    # Parameters to save
    params = {
        "model": args.model,
        "macro_type": args.macro_type,
        "target_type": args.target_type,
        "macro_weight": args.macro_weight,
        "rloss_weight": args.rloss_weight,
        "speed_loss": args.speed_loss,
        "masking": args.masking,
        "prev_out_aware": args.prev_out_aware,
        "bidirectional": args.bidirectional,
        "flip_pitch": args.flip_pitch,
        "n_features": args.n_features,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "cuda": args.cuda,
        "best_loss": args.best_loss,
    }

    # Hyperparameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    clip = args.clip
    print_every = args.print_every_batch
    save_every = args.save_every_epoch
    pretrain_time = args.pretrain_time

    # Set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model(args.model, params, parser).to(default_device)
    model = nn.DataParallel(model)

    # Update params with model parameters
    params = model.module.params
    params["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    save_path = "saved/{:03d}".format(args.trial)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/model")
    with open(f"{save_path}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Continue a previous experiment, or start a new one
    if args.cont:
        state_dict = torch.load("{}/model/{}_state_dict_best.pt".format(save_path, args.model))
        model.module.load_state_dict(state_dict)
    else:
        if args.model.endswith("lstm"):  # nonhierarchical models
            title = f"{args.trial} {args.target_type} | {args.model}"
        else:  # hierarchical models (team_ball or player_ball)
            title = f"{args.trial} {args.target_type} | {args.model}"
        if args.prev_out_aware:
            title += " | prev_out_aware"
        if args.bidirectional:
            title += " | bidirectional"

        print_keys = ["flip_pitch", "n_features", "batch_size", "start_lr"]
        if args.model in ["team_ball", "player_ball"]:
            print_keys += ["macro_weight"]
        if "rloss_weight" in params and params["rloss_weight"] > 0:
            print_keys += ["rloss_weight"]
        if "speed_loss" in params and params["speed_loss"]:
            print_keys += ["speed_loss"]
        if "masking" in params:
            print_keys += ["masking"]

        printlog(title)
        printlog(model.module.params_str)
        printlog(get_params_str(print_keys, model.module.params))
        printlog("n_params {:,}".format(params["total_params"]))
    printlog("############################################################")

    print()
    print("Generating datasets...")

    if args.target_type == "gk":
        train_files = ["match1.csv", "match2.csv", "match3_valid.csv"]
        valid_files = ["match3_test.csv"]

        train_paths = [f"data/metrica_traces/{f}" for f in train_files]
        valid_paths = [f"data/metrica_traces/{f}" for f in valid_files]

    else:  # if args.target_type == "ball":
        metrica_files = ["match1.csv", "match2.csv", "match3_valid.csv"]
        metrica_paths = [f"data/metrica_traces/{f}" for f in metrica_files]

        gps_files = os.listdir("data/gps_event_traces_gk_pred")
        gps_paths = [f"data/gps_event_traces_gk_pred/{f}" for f in gps_files]
        gps_paths.sort()

        assert args.train_fito or args.train_metrica
        train_paths = []
        if args.train_fito:
            train_paths += gps_paths[:10]
        if args.train_metrica:
            train_paths += metrica_paths[:-1]

        assert args.valid_fito or args.valid_metrica
        valid_paths = []
        if args.valid_fito:
            valid_paths += gps_paths[-5:-3]
        if args.valid_metrica:
            valid_paths += metrica_paths[-1:]

    macro_type = args.macro_type if args.model in ["team_ball", "player_ball"] else None
    nw = len(model.device_ids) * 4
    train_dataset = SoccerDataset(
        data_paths=train_paths,
        target_type=args.target_type,
        macro_type=macro_type,
        train=True,
        n_features=args.n_features,
        target_speed=args.speed_loss,
        flip_pitch=args.flip_pitch,
    )
    test_dataset = SoccerDataset(
        data_paths=valid_paths,
        target_type=args.target_type,
        macro_type=macro_type,
        train=False,
        n_features=args.n_features,
        target_speed=args.speed_loss,
        flip_pitch=args.flip_pitch,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    # Train loop
    best_sum_loss = args.best_loss
    best_mse_loss = args.best_loss
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for e in range(n_epochs):
        epoch = e + 1

        hyperparams = {"pretrain": epoch <= pretrain_time}

        # Set a custom learning rate schedule
        if epochs_since_best == 2 and lr > args.min_lr:
            # Load previous best model
            path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
            if epoch <= pretrain_time:
                path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
            state_dict = torch.load(path)

            # Decrease learning rate
            lr = max(lr / 2, args.min_lr)
            printlog("########## lr {} ##########".format(lr))
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(hyperparams_str(epoch, hyperparams))
        start_time = time.time()

        train_losses = run_epoch(model, optimizer, train=True, print_every=print_every)
        printlog("Train:\t" + loss_str(train_losses))

        test_losses = run_epoch(model, optimizer, train=False)
        printlog("Test:\t" + loss_str(test_losses))

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time))

        test_mse_loss = test_losses["mse_loss"]
        test_sum_loss = sum([value for key, value in test_losses.items() if key.endswith("loss")])

        # Best model on test set
        if best_sum_loss == 0 or test_sum_loss < best_sum_loss:
            best_sum_loss = test_sum_loss
            epochs_since_best = 0
            path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
            if epoch <= pretrain_time:
                path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
            torch.save(model.module.state_dict(), path)
            printlog("########## Best Model ###########")

        elif "mse_loss" in test_losses and test_losses["mse_loss"] < best_mse_loss:
            best_mse_loss = test_losses["mse_loss"]
            epochs_since_best = 0
            path = "{}/model/{}_state_dict_best_mse.pt".format(save_path, args.model)
            torch.save(model.module.state_dict(), path)
            printlog("######## Best MSE Model #########")

        # Periodically save model
        if epoch % save_every == 0:
            path = "{}/model/{}_state_dict_{}.pt".format(save_path, args.model, epoch)
            torch.save(model.module.state_dict(), path)
            printlog("########## Saved Model ##########")

        # End of pretrain stage
        if epoch == pretrain_time:
            printlog("######### End Pretrain ##########")
            best_sum_loss = 0
            epochs_since_best = 0
            lr = max(args.start_lr, args.min_lr)

            state_dict = torch.load("{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model))
            model.module.load_state_dict(state_dict)
            test_losses = run_epoch(model, optimizer, train=False)
            printlog("Test:\t" + loss_str(test_losses))

    printlog("Best Test Loss: {:.4f}".format(best_sum_loss))
