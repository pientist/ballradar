import torch
from torch.utils.data import Dataset


def get_random_datasets(b, k=4, min_size=100, max_size=500):
    """
    Generates a random mixture of k gaussians in the plane.
    Means are in square [-4, 4]x[-4, 4].
    Standard deviation is 0.3.

    Arguments:
        b: an integer, number of datasets.
        k: an integer, number of clusters in each dataset.
        min_size, max_size: integers.
    Returns:
        data: a float tensor with shape [b, n, 2],
            where min_size <= n <= max_size.
        params: a dict with the following keys
            'means': a float tensor with shape [b, k, 2],
            'variances': a float tensor with shape [b, k, 2],
            'pis': a float tensor with shape [b, k].
    """

    # dataset size
    n = torch.randint(min_size, max_size + 1, size=[], dtype=torch.int32)
    # shape []

    # parameters of gaussians
    centers = 4.0 * (2.0 * torch.rand(b, k, 2) - 1.0)
    sigmas = torch.Tensor([0.3]).view(1, 1, 1).repeat([b, k, 2])
    # they have shape [b, k, 2]

    # probabilities to be in different clusters
    pi = torch.distributions.Dirichlet(torch.Tensor(k*[1.0])).sample((b,))
    # shape [b, k]

    # assignments to each cluster
    labels = torch.distributions.categorical.Categorical(probs=pi).sample((n,)).t()
    # shape [b, n]

    labels = labels.unsqueeze(2).repeat([1, 1, 2])
    selected_centers = torch.gather(centers, 1, labels)
    selected_sigmas = torch.gather(sigmas, 1, labels)

    data = torch.normal(selected_centers, selected_sigmas)
    params = {'means': centers, 'variances': sigmas ** 2, 'pis': pi}
    return data, params
