import torch


def laplace_kernel(context_x, target_x, gamma=1):
    """
    Computes the Laplacian Kernel following the implemenation of sklearn
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/
    metrics/pairwise.py#L1076

    Parameters
    ----------
    context_x: tensor(batch_size, num_context, dimx): holds the context points

    target_x: tensor(batch_size, num_targets, dimx): holds the x values of the
              of the targets
    gamma: scaling facfor

    Returns: distance tensor (batch_size, num_targets, num_context, 1)
    -------
    """
    # batch wise distance taking--> for each context points calculate
    # the distance to each context point
    distances = context_x[:, None, :, :] - target_x[:, :, None, :]
    scaled_distance = - gamma * torch.abs(distances)
    exponentiated_distance = torch.exp(scaled_distance)
    # distance between entries is aggregated across dimensions
    aggregated_distance = exponentiated_distance.sum(-1)
    return aggregated_distance
