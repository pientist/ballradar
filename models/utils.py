import math

import torch


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        if arg.startswith("n_") or arg.endswith("_dim"):
            parser.add_argument("--" + arg, type=int, required=True)
        elif arg == "dropout":
            parser.add_argument("--" + arg, type=float, required=False, default=0)
    args, _ = parser.parse_known_args()

    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ""
    for arg in model_args:
        if arg in params:
            ret += " {} {} |".format(arg, params[arg])
    return ret[1:-2]


def cudafy_list(states, device):
    for i in range(len(states)):
        states[i] = states[i].to(device)
    return states


def index_by_agent(states, n_agents):
    x = states[1:, :, : 2 * n_agents].clone()
    x = x.view(x.size(0), x.size(1), n_agents, -1).transpose(1, 2)
    return x


def get_macro_ohe(macro, n_agents, M, device="cuda"):
    macro_ohe = torch.zeros(macro.size(0), n_agents, macro.size(1), M)
    for i in range(n_agents):
        macro_ohe[:, i, :, :] = one_hot_encode(macro[:, :, i].data, M)
    if macro.is_cuda:
        macro_ohe = macro_ohe.to(device)
    return macro_ohe


def sample_gauss(mean, std, device="cuda"):
    eps = torch.FloatTensor(std.size()).normal_().to(device)
    return eps.mul(std).add_(mean)


def nll_gauss(mean, std, x, device="cuda"):
    pi = torch.FloatTensor([math.pi]).to(device)
    nll_element = (x - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)
    return 0.5 * torch.sum(nll_element)


def kld_gauss(mean_1, std_1, mean_2, std_2):
    kld_element = (
        2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1
    )
    return 0.5 * torch.sum(kld_element)


def entropy_gauss(std, scale=1, device="cuda"):
    """Computes gaussian differential entropy."""
    pi, e = torch.FloatTensor([math.pi]).to(device), torch.FloatTensor([math.e]).to(device)
    return 0.5 * torch.sum(scale * torch.log(2 * pi * e * std))


def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).cpu().long()
    dims.append(N)
    ret = torch.zeros(dims)
    ret.scatter_(-1, inds, 1)
    return ret


def sample_multinomial(probs, device="cuda"):
    inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1)).to(device)
    return ret


def calc_speed(xy):
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    vx = torch.diff(x, prepend=x[:, [0]]) / 0.1
    vy = torch.diff(y, prepend=y[:, [0]]) / 0.1
    speed = torch.sqrt(vx**2 + vy**2 + torch.tensor(1e-6).to(xy.device))
    return torch.stack([x, y, speed], -1)


def calc_real_loss(pred_xy, input_features, n_features=6, eps=torch.tensor(1e-6)):
    eps = eps.to(pred_xy.device)

    # Calculate the angle between two consecutive velocity vectors
    # We skip the division by time difference, which is eventually reduced
    vels = pred_xy.diff(dim=1)
    speeds = torch.linalg.norm(vels, dim=-1)
    cos_num = torch.sum(vels[:, :-1] * vels[:, 1:], dim=-1) + eps
    cos_denom = speeds[:, :-1] * speeds[:, 1:] + eps
    cosines = torch.clamp(cos_num / cos_denom, -1 + eps, 1 - eps)
    angles = torch.acos(cosines)

    # Compute the distance between the ball and the nearest player
    pred_xy = torch.unsqueeze(pred_xy, dim=2)
    player_x = input_features[:, :, 0 : n_features * 22 : n_features]
    player_y = input_features[:, :, 1 : n_features * 22 : n_features]
    player_xy = torch.stack([player_x, player_y], dim=-1)
    ball_dists = torch.linalg.norm(pred_xy - player_xy, dim=-1)
    nearest_dists = torch.min(ball_dists, dim=-1).values[:, 1:-1]

    # Either course angle must be close to 0 or the ball must be close to a player
    return (torch.tanh(angles) * nearest_dists).mean()


def calc_trace_dist(pred_xy, target_xy, aggfunc="mean"):
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_acc(pred_poss, target_poss, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().sum().item()
