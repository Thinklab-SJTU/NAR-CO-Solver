import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple
from ortools.linear_solver import pywraplp
import torch_geometric as pyg
import time
from src.gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
from src.gumbel_linsat_layer import gumbel_linsat_layer, init_linsat_constraints
from constraint_layers.sinkhorn import Sinkhorn
from src.optimal_transport import ortools_ot, OptimalTransportLayer


####################################
#        helper functions          #
####################################

def compute_objective(points, selected_points, distance_metric='euclidean', demands=None):
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    dist_func = _get_distance_func(distance_metric)
    dist = dist_func(points, selected_points, points.device)
    if demands is None: # FLP w/o demand
        return torch.sum(torch.gather(dist, -1, torch.argmin(dist, dim=-1).unsqueeze(-1)).squeeze(-1), dim=-1)
    else: # FLP w/ demand
        if len(selected_points.shape) == 3:
            batch_size = selected_points.shape[0]
        else:
            selected_points = selected_points.unsqueeze(0)
            batch_size = 1
        dummy = selected_points.shape[1] - torch.sum(demands)
        row_dummy = torch.cat((demands, dummy.unsqueeze(-1)), dim=-1).unsqueeze(0).repeat(batch_size, 1)
        col = torch.ones(selected_points.shape[1]).unsqueeze(0).repeat(batch_size, 1)
        dist_dummy = torch.cat((dist, torch.zeros(batch_size, 1, dist.shape[2], device=dist.device)), dim=1)
        obj_scores = ortools_ot(dist_dummy, row_dummy, col)
        return obj_scores


def compute_objective_differentiable(dist, probs, demands=None, temp=30, sk_iter=40):
    if demands is None: # FLP w/o demand
        exp_dist = torch.exp(-temp / dist.mean() * dist)
        exp_dist_probs = exp_dist.unsqueeze(0) * probs.unsqueeze(-1)
        probs_per_dist = exp_dist_probs / exp_dist_probs.sum(1, keepdim=True)
        obj = (probs_per_dist * dist).sum(dim=(1, 2))
    else: # FLP w/ demand
        # the objective estimation is an OT problem: move probs to demands
        batch_size = probs.shape[0]
        # sk = Sinkhorn(max_iter=sk_iter, tau=1 / temp, batched_operation=True)
        sk2 = OptimalTransportLayer(gamma=temp, maxiters=sk_iter) # this implementation requires less GPU mem
        dummy = torch.sum(probs, dim=-1) - torch.sum(demands)
        col_dummy = torch.cat((demands.unsqueeze(0).repeat(batch_size, 1), dummy.unsqueeze(-1)), dim=-1)
        if len(dist.shape) == 2:
            dist_dummy = torch.cat((dist, torch.zeros(dist.shape[0], 1, device=dist.device)), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            assert len(dist.shape) == 3
            dist_dummy = torch.cat((dist, torch.zeros(dist.shape[0], dist.shape[1], 1, device=dist.device)), dim=-1)
        # transp_mat = sk(-dist_dummy, probs, col_dummy)
        transp_mat = sk2(dist_dummy, probs, col_dummy) * torch.sum(probs, dim=1).reshape(batch_size, 1, 1)
        obj = (transp_mat * dist_dummy).sum(dim=(1, 2))
    return obj


def build_graph_from_points(points, dist=None, return_dist=False, distance_metric='euclidean'):
    if dist is None:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    norm_dist = dist * 1.414 / dist.max()
    edge_indices = torch.nonzero(norm_dist <= 0.02, as_tuple=False).transpose(0, 1)
    edge_attrs = (points.unsqueeze(0) - points.unsqueeze(1))[torch.nonzero(norm_dist <= 0.02, as_tuple=True)] + 0.5
    g = pyg.data.Data(x=points, edge_index=edge_indices, edge_attr=edge_attrs)
    if return_dist:
        return g, dist
    else:
        return g


#################################################
#             Learning FLP Methods              #
#################################################

class GNNModel(torch.nn.Module):
    # clustering model (3-layer SplineConv)
    def __init__(self):
        super(GNNModel, self).__init__()
        self.gconv1 = pyg.nn.SplineConv(2, 16, 2, 5)
        self.gconv2 = pyg.nn.SplineConv(16, 16, 2, 5)
        self.gconv3 = pyg.nn.SplineConv(16, 16, 2, 5)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, g):
        x = torch.relu(self.gconv1(g.x, g.edge_index, g.edge_attr))
        x = torch.relu(self.gconv2(x, g.edge_index, g.edge_attr))
        x = torch.relu(self.gconv3(x, g.edge_index, g.edge_attr))
        x = self.fc(x).squeeze(-1)
        return torch.sigmoid(x)

    def zero_params(self):
        for param in self.parameters():
            param.zero_()


def cardnn_facility_location(points, num_facilities, model,
                             softmax_temp, sample_num, noise, tau, sk_iters, opt_iters,
                             grad_step=0.1, time_limit=-1, distance_metric='euclidean', verbose=True):
    """
    The Cardinality neural network (CardNN) solver for facility location. This implementation supports 3 variants:
    CardNN-S (Sinkhorn), CardNN-GS (Gumbel-Sinkhorn) and CardNN-HGS (Homotopy-Gumbel-Sinkhorn).

    Args:
        points: the set of points
        num_facilities: max number of facilities (i.e. the cardinality)
        model: the GNN model
        softmax_temp: temperature of softmax (actually softmin) when estimating the objective
        sample_num: sampling number of Gumbel
        noise: sigma of Gumbel noise
        tau: annealing parameter of Sinkhorn
        sk_iters: number of max iterations of Sinkhorn
        opt_iters: number of optimizaiton operations in testing
        grad_step: the gradient step (i.e. "learning rate") in testing-time optimization
        time_limit: upper limit of solving time
        distance_metric: euclidean or manhattan
        verbose: show more information in solving

    If noise=0, it is CardNN-S;
    If noise>0, it is CardNN-GS;
    If multiple values of "noise, tau, sk_iters, opt_iters" are given, it is CardNN-HGS.

    Returns: best objective score, best solution
    """
    prev_time = time.time()

    # Graph modeling of the original problem
    graph, dist = build_graph_from_points(points, None, True, distance_metric)

    # Predict the initial latent variables by NN
    latent_vars = model(graph).detach()

    # Optimize over the latent variables in test
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=grad_step)
    best_obj = float('inf')
    best_top_k_indices = []
    best_found_at_idx = -1

    # Optimization steps (by gradient)
    if type(noise) == list and type(tau) == list and type(sk_iters) == list and type(opt_iters) == list:
        iterable = zip(noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([noise], [tau], [sk_iters], [opt_iters])
    opt_iter_offset = 0
    for noise, tau, sk_iters, opt_iters in iterable:
        for opt_idx in range(opt_iter_offset, opt_iter_offset + opt_iters):
            # time limit control
            if time_limit > 0 and time.time() - prev_time > time_limit:
                break

            gumbel_weights_float = torch.sigmoid(latent_vars)
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float, num_facilities,
                max_iter=sk_iters, tau=tau, sample_num=sample_num, noise_fact=noise, return_prob=True
            )

            # estimate objective by softmax (in the computational graph)
            obj = compute_objective_differentiable(dist, probs, temp=softmax_temp)

            obj.mean().backward()
            if opt_idx % 10 == 0 and verbose:
                print(f'idx:{opt_idx} estimated {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}')

            # compute the real objective (detached from the computational graph)
            cluster_centers = torch.gather(
                torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                1,
                torch.repeat_interleave(top_k_indices.unsqueeze(-1), points.shape[-1], -1)
            )
            obj = compute_objective(points.unsqueeze(0), cluster_centers, distance_metric)

            # find the best solution till now
            best_idx = torch.argmin(obj)
            min_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            if min_obj < best_obj:
                best_obj = min_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = opt_idx
            if opt_idx % 10 == 0 and verbose:
                print(f'idx:{opt_idx} real {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}, now time:{time.time()-prev_time:.2f}')
            optimizer.step()
            optimizer.zero_grad()
        opt_iter_offset += opt_iters
    cluster_centers = torch.stack([torch.gather(points[:, _], 0, best_top_k_indices) for _ in range(points.shape[1])], dim=-1)

    # fast neighbor search by k-means (as post-processing)
    choice_cluster, cluster_centers, selected_indices = discrete_kmeans(points, num_facilities, init_x=cluster_centers, distance=distance_metric, device=points.device)
    objective = compute_objective(points, cluster_centers, distance_metric).item()

    return objective, selected_indices, time.time() - prev_time


def linsat_facility_location(points, num_facilities, demands, model,
                             softmax_temp, sample_num, noise, tau, sk_iters, opt_iters,
                             grad_step=0.1, time_limit=-1, distance_metric='euclidean', verbose=True):
    """
    The neural network solver for facility location based on LinSATNet.

    Args:
        points: the set of points
        num_facilities: max number of facilities (i.e. the cardinality). If not None, we are solving K-Median FLP
        demands: demands of each point. If not None, we are solving standard FLP
        model: the GNN model
        softmax_temp: temperature of softmax (actually softmin) when estimating the objective
        sample_num: sampling number of Gumbel
        noise: sigma of Gumbel noise
        tau: annealing parameter of LinSAT
        sk_iters: number of max iterations of LinSAT
        opt_iters: number of optimizaiton operations in testing
        grad_step: the gradient step (i.e. "learning rate") in testing-time optimization
        time_limit: upper limit of solving time
        distance_metric: euclidean or manhattan
        verbose: show more information in solving

    Returns: best objective score, best solution
    """
    prev_time = time.time()

    # Graph modeling of the original problem
    graph, dist = build_graph_from_points(points, None, True, distance_metric)

    # Predict the initial latent variables by NN
    latent_vars = model(graph).detach()

    # Optimize over the latent variables in test
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=grad_step)
    best_obj = float('inf')
    best_top_k_indices = []
    best_found_at_idx = -1

    # Optimization steps (by gradient)
    if type(softmax_temp) == list and type(noise) == list and type(tau) == list and type(sk_iters) == list and type(opt_iters) == list:
        iterable = zip(softmax_temp, noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([softmax_temp], [noise], [tau], [sk_iters], [opt_iters])
    opt_iter_offset = 0
    linsat_constr = None
    for softmax_temp, noise, tau, sk_iters, opt_iters in iterable:
        for opt_idx in range(opt_iter_offset, opt_iter_offset + opt_iters):
            gumbel_weights_float = torch.sigmoid(latent_vars)
            # time limit control
            if time_limit > 0 and time.time() - prev_time > time_limit:
                break

            A = torch.ones(1, len(points), device=latent_vars.device)
            b = torch.tensor([num_facilities], dtype=A.dtype, device=A.device)
            if demands is None:
                C = d = None
            else:
                C = torch.ones(1, len(points), device=latent_vars.device)
                d = torch.tensor([torch.sum(demands)], dtype=C.dtype, device=latent_vars.device)
            if linsat_constr is None:
                linsat_constr = init_linsat_constraints(gumbel_weights_float, A, b, C, d)
            probs = gumbel_linsat_layer(gumbel_weights_float, linsat_constr,
                                        max_iter=sk_iters, tau=tau, sample_num=sample_num, noise_fact=noise)

            # estimate objective by softmax (in the computational graph)
            obj = compute_objective_differentiable(dist, probs, demands, temp=softmax_temp, sk_iter=40)

            obj.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

            if opt_idx % 10 == 0 and verbose:
                print(f'idx:{opt_idx} estimated {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}')

            # compute the real objective (detached from the computational graph)
            top_k_indices = torch.topk(probs, num_facilities, dim=-1).indices
            if demands is None:
                selected_points = torch.gather(
                    torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                    1,
                    torch.repeat_interleave(top_k_indices.unsqueeze(-1), points.shape[-1], -1)
                )
                obj = compute_objective(points.unsqueeze(0), selected_points, distance_metric, demands)
            else: # use faster sinkhorn OT
                dist_to_selected = torch.gather(
                    torch.repeat_interleave(dist.unsqueeze(0), top_k_indices.shape[0], 0),
                    1,
                    torch.repeat_interleave(top_k_indices.unsqueeze(-1), dist.shape[-1], -1)
                )
                obj = compute_objective_differentiable(
                    dist_to_selected,
                    torch.ones(sample_num, num_facilities, device=probs.device),
                    demands,
                    temp=1000, sk_iter=100
                )

                # slower version, for reference
                # selected_points = torch.gather(
                #     torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                #     1,
                #     torch.repeat_interleave(top_k_indices.unsqueeze(-1), points.shape[-1], -1)
                # )
                # obj = compute_objective(points.unsqueeze(0), selected_points, distance_metric, demands)

            best_idx = torch.argmin(obj)
            top_k_indices = top_k_indices[best_idx]
            selected_points = torch.stack(
                [torch.gather(points[:, _], 0, top_k_indices) for _ in range(points.shape[1])], dim=-1)
            # fast neighborhood search by k-means
            choice_cluster, selected_points, selected_indices = discrete_kmeans(
                points, num_facilities, init_x=selected_points, distance=distance_metric, device=points.device)
            if demands is None:
                min_obj = compute_objective(points.unsqueeze(0), selected_points, distance_metric, demands).item()
            else:
                dist_to_selected = torch.gather(dist, 0, torch.repeat_interleave(top_k_indices.unsqueeze(-1), dist.shape[-1], -1))
                min_obj = compute_objective_differentiable(
                    dist_to_selected.unsqueeze(0),
                    torch.ones(1, num_facilities, device=probs.device),
                    demands,
                    temp=1000, sk_iter=100
                ).item()

            # find the best solution till now
            if min_obj < best_obj:
                best_obj = min_obj
                best_top_k_indices = selected_indices
                best_found_at_idx = opt_idx
            if opt_idx % 10 == 0 and verbose:
                print(f'idx:{opt_idx} real {min_obj:.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}, now time:{time.time()-prev_time:.2f}')

        opt_iter_offset += opt_iters

    if demands is not None:
        selected_points = torch.stack([torch.gather(points[:, _], 0, best_top_k_indices)
                                       for _ in range(points.shape[1])], dim=-1)
        best_obj = compute_objective(points.unsqueeze(0), selected_points, distance_metric, demands).item()
    return best_obj, best_top_k_indices, time.time() - prev_time


def egn_facility_location(points, num_facilities, model,
                          softmax_temp, egn_beta, random_trials=0,
                          time_limit=-1, distance_metric='euclidean'):
    """
    Our implementation of the Erdos Goes Neural (EGN) solver for facility location.
    """
    prev_time = time.time()
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    graph.ori_x = graph.x.clone()
    best_objective = float('inf')
    best_selected_indices = None
    for _ in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0:
            graph.x = graph.ori_x + torch.randn_like(graph.x) / 100
        probs = model(graph).detach()
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        for prob_idx in probs_argsort:
            if selected_items >= num_facilities:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            constraint_conflict_0 = torch.relu(probs_0.sum() - num_facilities)
            constraint_conflict_1 = torch.relu(probs_1.sum() - num_facilities)
            obj_0 = compute_objective_differentiable(dist, probs_0,
                                                     temp=softmax_temp) + egn_beta * constraint_conflict_0
            obj_1 = compute_objective_differentiable(dist, probs_1,
                                                     temp=softmax_temp) + egn_beta * constraint_conflict_1
            if obj_0 >= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = torch.topk(probs, num_facilities, dim=-1).indices
        cluster_centers = torch.stack([torch.gather(points[:, _], 0, top_k_indices) for _ in range(points.shape[1])],
                                      dim=-1)
        choice_cluster, cluster_centers, selected_indices = discrete_kmeans(points, num_facilities,
                                                                            init_x=cluster_centers,
                                                                            distance=distance_metric,
                                                                            device=points.device)
        objective = compute_objective(points, cluster_centers, distance_metric).item()
        if objective < best_objective:
            best_objective = objective
            best_selected_indices = selected_indices
    return best_objective, best_selected_indices, time.time() - prev_time


#################################################
#            Traditional FLP Methods            #
#################################################

def initialize(X: Tensor, num_clusters: int, method: str='plus') -> np.array:
    r"""
    Initialize cluster centers.

    :param X: matrix
    :param num_clusters: number of clusters
    :param method: denotes different initialization strategies: ``'plus'`` (default) or ``'random'``
    :return: initial state

    .. note::
        We support two initialization strategies: random initialization by setting ``method='random'``, or `kmeans++
        <https://en.wikipedia.org/wiki/K-means%2B%2B>`_ by setting ``method='plus'``.
    """
    if method == 'plus':
        init_func = _initialize_plus
    elif method == 'random':
        init_func = _initialize_random
    else:
        raise NotImplementedError
    return init_func(X, num_clusters)


def _initialize_random(X, num_clusters):
    """
    Initialize cluster centers randomly. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def _initialize_plus(X, num_clusters):
    """
    Initialize cluster centers by k-means++. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    centroid_index = np.zeros(num_clusters)
    for i in range(num_clusters):
        if i == 0:
            choice_prob = np.full(num_samples, 1 / num_samples)
        else:
            centroid_X = X[centroid_index[:i]]
            dis = _pairwise_euclidean(X, centroid_X)
            dis_to_nearest_centroid = torch.min(dis, dim=1).values
            choice_prob = dis_to_nearest_centroid / torch.sum(dis_to_nearest_centroid)
            choice_prob = choice_prob.detach().cpu().numpy()

        centroid_index[i] = np.random.choice(num_samples, 1, p=choice_prob, replace=False)

    initial_state = X[centroid_index]
    return initial_state


def discrete_kmeans(
        X: Tensor,
        num_clusters: int,
        init_x: Union[Tensor, str]='plus',
        distance: str='euclidean',
        tol: float=1e-4,
        device=torch.device('cpu'),
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform discrete kmeans on given data matrix :math:`\mathbf X`.
    Here "discrete" means the selected cluster centers must be a subset of the input data :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: convergence threshold [default: 0.0001]
    :param device: computing device [default: cpu]
    :return: cluster ids, cluster centers, selected indices
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if init_x == 'rand':
        initial_state = X[torch.randperm(X.shape[0])[:num_clusters], :]
    elif type(init_x) is str:
        initial_state = initialize(X, num_clusters, method=init_x)
    else:
        initial_state = init_x

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state, device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        selected_indices = torch.zeros(num_clusters, device=device, dtype=torch.long)
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze(-1)
            selected_X = torch.index_select(X, 0, selected)
            intra_selected_dist = pairwise_distance_function(selected_X, selected_X, device)
            index_for_selected = torch.argmin(intra_selected_dist.sum(dim=1))
            initial_state[index] = selected_X[index_for_selected]
            selected_indices[index] = selected[index_for_selected]

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1
        if center_shift ** 2 < tol:
            break
        if torch.isnan(initial_state).any():
            print('NAN encountered in clustering. Retrying...')
            initial_state = initialize(X, num_clusters)

    return choice_cluster, initial_state, selected_indices


def kmeans(
        X: Tensor,
        num_clusters: int,
        weight: Tensor=None,
        init_x: Union[Tensor, str]='plus',
        distance: str='euclidean',
        tol: float=1e-4,
        device=torch.device('cpu'),
) -> Tuple[Tensor, Tensor]:
    r"""
    Perform kmeans on given data matrix :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: convergence threshold [default: 0.0001]
    :param device: computing device [default: cpu]
    :return: cluster ids, cluster centers
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(init_x) is str:
        initial_state = initialize(X, num_clusters, method=init_x)
    else:
        initial_state = init_x

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state, device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increase iteration
        iteration = iteration + 1
        if center_shift ** 2 < tol:
            break
        if torch.isnan(initial_state).any():
            print('NAN encountered in clustering. Retrying...')
            initial_state = initialize(X, num_clusters)

    return choice_cluster, initial_state


def kmeans_predict(
        X: Tensor,
        cluster_centers: Tensor,
        weight: Tensor=None,
        distance: str='euclidean',
        device=torch.device('cpu')
) -> Tensor:
    r"""
    Kmeans prediction using existing cluster centers.

    :param X: matrix
    :param cluster_centers: cluster centers
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: computing device [default: 'cpu']
    :return: cluster ids
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers, device)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def _get_distance_func(distance):
    if distance == 'euclidean':
        return _pairwise_euclidean
    elif distance == 'cosine':
        return _pairwise_cosine
    elif distance == 'manhattan':
        return _pairwise_manhattan
    else:
        raise NotImplementedError


def _pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    """Compute pairwise Euclidean distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def _pairwise_manhattan(data1, data2, device=torch.device('cpu')):
    """Compute pairwise Manhattan distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = torch.abs(A - B)
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def _pairwise_cosine(data1, data2, device=torch.device('cpu')):
    """Compute pairwise cosine distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze(-1)
    return cosine_dis


def spectral_clustering(sim_matrix: Tensor, cluster_num: int, init: Tensor=None,
                        return_state: bool=False, normalized: bool=False):
    r"""
    Perform spectral clustering based on given similarity matrix.

    This function firstly computes the leading eigenvectors of the given similarity matrix, and then utilizes the
    eigenvectors as features and performs k-means clustering based on these features.

    :param sim_matrix: :math:`(n\times n)` input similarity matrix. :math:`n`: number of instances
    :param cluster_num: number of clusters
    :param init: the initialization technique or initial features for k-means
    :param return_state: whether return state features (can be further used for prediction)
    :param normalized: whether to normalize the similarity matrix by its degree
    :return: the belonging of each instance to clusters, state features (if ``return_state==True``)
    """
    degree = torch.diagflat(torch.sum(sim_matrix, dim=-1))
    if normalized:
        aff_matrix = (degree - sim_matrix) / torch.diag(degree).unsqueeze(1)
    else:
        aff_matrix = degree - sim_matrix
    e, v = torch.symeig(aff_matrix, eigenvectors=True)
    topargs = torch.argsort(torch.abs(e), descending=False)[1:cluster_num]
    v = v[:, topargs]

    if cluster_num == 2:
        choice_cluster = (v > 0).to(torch.int).squeeze(1)
    else:
        choice_cluster, initial_state = kmeans(v, cluster_num, init_x=init if init is not None else 'plus',
                                               distance='euclidean', tol=1e-6)

    choice_cluster = choice_cluster.to(sim_matrix.device)

    if return_state:
        return choice_cluster, initial_state
    else:
        return choice_cluster


def greedy_facility_location(
        X: Tensor,
        num_facilities: int,
        demands: Tensor=None,
        distance: str='euclidean',
        device=torch.device('cpu'),
        sk_iter = 200,
        sk_temp = 200,
) -> Tuple[Tensor, Tensor]:
    r"""
    Greedy algorithm for facility location problem.
    This is function also solves the discrete clustering problem given data matrix :math:`\mathbf X`.
    Here "discrete" means the selected cluster centers must be a subset of the input data :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_facilities: (int) number of facilities to be selected
    :param demands: (optional) the demand on each point
    :param distance: distance [options: 'euclidean', 'manhattan', 'cosine'] [default: 'euclidean']
    :param device: computing device [default: cpu]
    :param sk_iter: number of iterations of Sinkhorn (useful when demands is not None)
    :param sk_temp: temperature parameter of Sinkhorn (useful when demands is not None)
    :return: cluster centers, selected indices
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    selected_indices = []
    unselected_indices = list(range(X.shape[0]))
    if demands is not None:
        sk = Sinkhorn(max_iter=sk_iter, tau=1 / sk_temp, batched_operation=True)
    for cluster_center_idx in range(num_facilities):
        best_obj = float('inf')
        best_idx = -1
        for unselected_idx in unselected_indices:
            selected = torch.tensor(selected_indices + [unselected_idx], device=device)
            selected_X = torch.index_select(X, 0, selected)
            dis = pairwise_distance_function(X, selected_X, device)
            if demands is None:
                obj = dis.min(dim=1).values.sum()
            else:
                if selected_X.shape[0] >= torch.sum(demands):
                    #obj = compute_objective(X, selected_X, distance, demands)
                    dummy = selected_X.shape[0] - torch.sum(demands)
                    row = torch.cat((demands, torch.tensor([dummy], device=device)))
                    col = torch.ones(selected_X.shape[0], device=device)
                    dis = torch.cat((dis, torch.zeros(1, dis.shape[1], device=device)), dim=0)
                else:
                    dummy = torch.sum(demands) - selected_X.shape[0]
                    col = torch.cat((torch.ones(selected_X.shape[0], device=device),
                                     torch.tensor([dummy], device=device)))
                    row = demands
                    dis = torch.cat((dis, torch.zeros(dis.shape[0], 1, device=device)), dim=1)
                    #obj = ortools_ot(dis_dummy, demands, col)
                transp_mat = sk(-dis.unsqueeze(0), row.unsqueeze(0), col.unsqueeze(0)).squeeze(0)
                obj = (transp_mat * dis).sum()

            if obj < best_obj:
                best_obj = obj
                best_idx = unselected_idx

        unselected_indices.remove(best_idx)
        selected_indices.append(best_idx)

    selected_indices = torch.tensor(selected_indices, device=device)
    cluster_centers = torch.index_select(X, 0, selected_indices)
    return cluster_centers, selected_indices


def ortools_facility_location(
        X: Tensor,
        num_facilities: int,
        distance: str='euclidean',
        solver_name=None,
        linear_relaxation=True,
        timeout_sec=60,
):
    # define solver instance
    if solver_name is None:
        if linear_relaxation:
            solver = pywraplp.Solver('facility_location',
                                     pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        else:
            solver = pywraplp.Solver('facility_location',
                                     pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    else:
        solver = pywraplp.Solver.CreateSolver(solver_name)

    X = X.cpu()

    # Initialize variables
    VarX = {}
    VarY = {}
    ConstY1 = {}
    ConstY2 = {}
    for selected_id in range(X.shape[0]):
        if linear_relaxation:
            VarX[selected_id] = solver.NumVar(0.0, 1.0, f'x_{selected_id}')
        else:
            VarX[selected_id] = solver.BoolVar(f'x_{selected_id}')

        VarY[selected_id] = {}
        for all_point_id in range(X.shape[0]):
            if linear_relaxation:
                VarY[selected_id][all_point_id] = solver.NumVar(0.0, 1.0, f'y_{selected_id}_{all_point_id}')
            else:
                VarY[selected_id][all_point_id] = solver.BoolVar(f'y_{selected_id}_{all_point_id}')

    # Constraints
    X_constraint = 0
    for selected_id in range(X.shape[0]):
        X_constraint += VarX[selected_id]
    solver.Add(X_constraint <= num_facilities)

    for selected_id in range(X.shape[0]):
        ConstY1[selected_id] = 0
        for all_point_id in range(X.shape[0]):
            ConstY1[selected_id] += VarY[selected_id][all_point_id]
        solver.Add(ConstY1[selected_id] <= VarX[selected_id] * X.shape[0])

    for all_point_id in range(X.shape[0]):
        ConstY2[all_point_id] = 0
        for selected_id in range(X.shape[0]):
            ConstY2[all_point_id] += VarY[selected_id][all_point_id]
        solver.Add(ConstY2[all_point_id] == 1)

    # The distance
    pairwise_distance_function = _get_distance_func(distance)
    distance_matrix = pairwise_distance_function(X, X)

    # the objective
    distance = 0
    for selected_id in range(X.shape[0]):
        for all_point_id in range(X.shape[0]):
            distance += distance_matrix[selected_id][all_point_id].item() * VarY[selected_id][all_point_id]

    solver.Minimize(distance)

    if timeout_sec > 0:
        solver.set_time_limit(int(timeout_sec * 1000))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(X.shape[0])]
    else:
        print('The problem does not have an optimal solution. status={}.'.format(status))
        return solver.Objective().Value(), [VarX[_].solution_value() for _ in range(X.shape[0])]


def gurobi_facility_location(
        X: Tensor,
        num_facilities: int,
        demands: Tensor=None,
        distance: str='euclidean',
        linear_relaxation=True,
        timeout_sec=60,
        start=None,
        verbose=True
):
    import gurobipy as grb
    try:
        model = grb.Model('facility location')
        if verbose:
            model.setParam('LogToConsole', 1)
        else:
            model.setParam('LogToConsole', 0)
        #model.setParam('MIPFocus', 1)
        if timeout_sec > 0:
            model.setParam('TimeLimit', timeout_sec)

        X = X.cpu()

        # Initialize variables
        VarX = {} # decision variable: select/not select
        VarY = {} # auxiliary variable: y_{i,j} is moved from i to j
        ConstY1 = {}
        ConstY2 = {}
        for selected_id in range(X.shape[0]):
            if linear_relaxation:
                VarX[selected_id] = model.addVar(0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f'x_{selected_id}')
            else:
                VarX[selected_id] = model.addVar(vtype=grb.GRB.BINARY, name=f'x_{selected_id}')
            if start is not None:
                VarX[selected_id].start = start[selected_id]
            VarY[selected_id] = {}
            for all_point_id in range(X.shape[0]):
                if linear_relaxation or demands is not None:
                    VarY[selected_id][all_point_id] = model.addVar(0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f'y_{selected_id}_{all_point_id}')
                else:
                    VarY[selected_id][all_point_id] = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{selected_id}_{all_point_id}')

        # Constraints
        X_constraint = 0
        for selected_id in range(X.shape[0]):
            X_constraint += VarX[selected_id]
        model.addConstr(X_constraint <= num_facilities)
        for selected_id in range(X.shape[0]):
            ConstY1[selected_id] = 0
            for all_point_id in range(X.shape[0]):
                ConstY1[selected_id] += VarY[selected_id][all_point_id]
            if demands is None:
                model.addConstr(ConstY1[selected_id] <= VarX[selected_id] * X.shape[0])
            else:
                model.addConstr(ConstY1[selected_id] <= VarX[selected_id])
        for all_point_id in range(X.shape[0]):
            ConstY2[all_point_id] = 0
            for selected_id in range(X.shape[0]):
                ConstY2[all_point_id] += VarY[selected_id][all_point_id]
            if demands is None:
                model.addConstr(ConstY2[all_point_id] == 1)
            else:
                model.addConstr(ConstY2[all_point_id] == demands[all_point_id])

        # The distance
        pairwise_distance_function = _get_distance_func(distance)
        distance_matrix = pairwise_distance_function(X, X)

        # the objective
        distance = 0
        for selected_id in range(X.shape[0]):
            for all_point_id in range(X.shape[0]):
                distance += distance_matrix[selected_id][all_point_id].item() * VarY[selected_id][all_point_id]
        model.setObjective(distance, grb.GRB.MINIMIZE)

        model.optimize()

        res = [model.getVarByName(f'x_{set_id}').X for set_id in range(X.shape[0])]
        if linear_relaxation:
            try:
                res = np.array(res, dtype=np.float)
            except:
                res = np.array(res, dtype=float)
        else:
            try:
                res = np.array(res, dtype=np.int)
            except:
                res = np.array(res, dtype=int)
        return model.getObjective().getValue(), torch.from_numpy(res).to(X.device)

    except grb.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
