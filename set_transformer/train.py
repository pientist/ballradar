import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from input_pipeline import get_random_datasets
from model import SetTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 1024
NUM_STEPS = 20000
EVAL_STEP = 1000
SAVE_PREFIX = "models/run00"
LOGS_DIR = "summaries/run00/"
DEVICE = torch.device("cuda:0")
USE_FLOAT16 = True

K = 4  # number of components
MIN_SIZE = 100  # minimal number of points
MAX_SIZE = 500  # maximal number of points

if USE_FLOAT16:
    from apex import amp


class LogLikelihood(nn.Module):
    """
    The log-likelihood of a dataset X = {x_1, ..., x_n}
    generated from a Mixture of Gaussians with k components.
    All gaussians are assumed to have diagonal variance matrices.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, means, variances, pis):
        """
        This function returns negative
        log-likelihood averaged per data.
        Also it averages over the batch dimension.

        `x` - a batch of datasets of equal size.
        `means, variances, pis` - parameters of distributions.

        Arguments:
            x: a float tensor with shape [b, n, c].
            means: a float tensor with shape [b, k, c].
            variances: a float tensor with shape [b, k, c],
                has positive values only.
            pis: a float tensor with shape [b, k],
                has positive values only.
        Returns:
            a float tensor with shape [].
        """
        device = x.device
        c = x.size(2)

        EPSILON = torch.tensor(1e-8, device=device)
        PI = torch.tensor(3.141592653589793, device=device)

        variances = variances + EPSILON
        pis = pis + EPSILON

        x = x.unsqueeze(2)  # shape [b, n, 1, c]
        means = means.unsqueeze(1)  # shape [b, 1, k, c]
        variances = variances.unsqueeze(1)  # shape [b, 1, k, c]
        pis = pis.unsqueeze(1)  # shape [b, 1, k]

        x = x - means  # shape [b, n, k, c]
        x = -0.5 * c * torch.log(2.0 * PI) - 0.5 * variances.log().sum(3) - 0.5 * (x.pow(2) / variances).sum(3)
        # it has shape [b, n, k], it represents log likelihood of multivariate normal distribution

        x = x + pis.log()
        # it has shape [b, n, k]

        average_likelihood = x.logsumexp(2).mean(1)
        # it has shape [b]

        # now average over the batch
        return average_likelihood.mean(0).neg()


def get_parameters(y):
    """
    This function transforms output of the network
    to the parameters of the distribution.
    """

    b = y.size(0)  # batch size
    y = torch.split(y, [2 * K, 2 * K, K], dim=1)

    means = y[0].view(b, K, 2)
    variances = y[1].exp().view(b, K, 2)
    pis = F.softmax(y[2], dim=1)  # shape [b, K]

    return means, variances, pis


def compute_groundtruth(x, params, criterion):
    """
    This function computes negative
    log-likelihood (average per data)
    using true distribution parameters.
    """
    params = {k: v.to(x.device) for k, v in params.items()}
    loss = criterion(x, params["means"], params["variances"], params["pis"])
    return loss


def train_and_evaluate():

    criterion = LogLikelihood()
    val_datasets = []
    num_val_batches = 500
    true_val_loss = 0.0

    for _ in range(num_val_batches):

        x, params = get_random_datasets(BATCH_SIZE, K, MIN_SIZE, MAX_SIZE)
        val_datasets.append(x)

        loss = compute_groundtruth(x, params, criterion)
        true_val_loss += loss.item()

    true_val_loss /= num_val_batches

    writer = SummaryWriter(LOGS_DIR)
    model = SetTransformer(in_dim=2, out_dim=5 * K)
    model = model.train().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=1e-4)

    if USE_FLOAT16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for iteration in range(1, NUM_STEPS + 1):

        x, params = get_random_datasets(BATCH_SIZE, K, MIN_SIZE, MAX_SIZE)
        # note that each iteration the datasets have different size

        x = x.to(DEVICE)
        # it has shape [b, n, 2],
        # where n is the dataset size

        start_time = time.perf_counter()
        optimizer.zero_grad()

        y = model(x)  # shape [b, 5 * K]
        means, variances, pis = get_parameters(y)
        loss = criterion(x, means, variances, pis)

        true_loss = compute_groundtruth(x, params, criterion)
        difference = loss - true_loss

        if USE_FLOAT16:
            with amp.scale_loss(loss, optimizer) as loss_scaled:
                loss_scaled.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        step_time = time.perf_counter() - start_time
        step_time = round(1000 * step_time, 1)

        writer.add_scalar("step_time", step_time, iteration)
        writer.add_scalar("loss", loss.item(), iteration)
        writer.add_scalar("difference_with_true_loss", difference.item(), iteration)
        print(f"iteration {iteration}, time {step_time} ms, {loss.item():.3f}")

        if iteration % EVAL_STEP == 0:
            loss = evaluate(model, criterion, val_datasets)
            writer.add_scalar("val_loss", loss, iteration)
            writer.add_scalar("difference_with_true_val_loss", loss - true_val_loss, iteration)
            path = f"{SAVE_PREFIX}_iteration_{iteration}.pth"
            torch.save(model.state_dict(), path)


def evaluate(model, criterion, val_datasets):

    model.eval()
    total_loss = 0.0

    for x in val_datasets:

        with torch.no_grad():

            x = x.to(DEVICE)
            y = model(x)

            means, variances, pis = get_parameters(y)
            loss = criterion(x, means, variances, pis)

        total_loss += loss.item()

    model.train()
    num_samples = len(val_datasets)
    return total_loss / num_samples


if __name__ == "__main__":
    train_and_evaluate()
