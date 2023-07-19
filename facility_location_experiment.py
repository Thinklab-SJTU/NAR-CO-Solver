from src.facility_location_methods import *
import time
import xlwt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
from src.facility_location_data import get_random_data, get_starbucks_data
from src.config import load_config

####################################
#             config               #
####################################

cfg = load_config()
device = torch.device('cuda:0')

wb = xlwt.Workbook()
ws = wb.add_sheet(f'clustering_{cfg.test_data_type}_{cfg.test_num_centers}-{cfg.num_data}')
ws.write(0, 0, 'name')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


####################################
#            training              #
####################################

if cfg.train_data_type == 'random':
    train_dataset = get_random_data(cfg.num_data, cfg.dim, 0, device)
elif cfg.train_data_type == 'starbucks':
    train_dataset = get_starbucks_data(device)
else:
    raise ValueError(f'Unknown dataset name {cfg.train_dataset_type}!')

model = GNNModel().to(device)

for method_name in cfg.methods:
    model_path = f'facility_location_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_{method_name}.pt'
    if not os.path.exists(model_path) and method_name in ['cardnn-gs', 'cardnn-s', 'egn']:
        print(f'Training the model weights for {method_name}...')
        model = GNNModel().to(device)
        if method_name in ['cardnn-gs', 'cardnn-s']:
            train_outer_optimizer = torch.optim.Adam(model.parameters(), lr=.1)
            for epoch in range(cfg.train_iter):
                # training loop
                obj_sum = 0
                for index, (_, points) in enumerate(train_dataset):
                    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
                    latent_vars = model(graph)
                    if method_name == 'cardnn-gs':
                        sample_num = cfg.gumbel_sample_num
                        noise_fact = cfg.gumbel_sigma
                    else:
                        sample_num = 1
                        noise_fact = 0
                    top_k_indices, probs = gumbel_sinkhorn_topk(
                        latent_vars, cfg.train_num_centers, max_iter=100, tau=.05,
                        sample_num=sample_num, noise_fact=noise_fact, return_prob=True
                    )
                    # compute objective by softmax
                    obj = compute_objective_differentiable(dist, probs, temp=50) # set smaller temp during training
                    obj.mean().backward()
                    obj_sum += obj.mean()
                    train_outer_optimizer.step()
                    train_outer_optimizer.zero_grad()
                print(f'epoch {epoch}/{cfg.train_iter}, obj={obj_sum / len(train_dataset)}')
        if method_name in ['egn']:
            train_outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # training loop
            for epoch in range(cfg.train_iter_egn):
                obj_sum = 0
                for index, (_, points) in enumerate(train_dataset):
                    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
                    probs = model(graph)
                    constraint_conflict = torch.relu(probs.sum() - cfg.train_num_centers)
                    obj = compute_objective_differentiable(dist, probs, temp=50) + cfg.egn_beta * constraint_conflict
                    obj.mean().backward()
                    obj_sum += obj.mean()
                    train_outer_optimizer.step()
                    train_outer_optimizer.zero_grad()

                print(f'epoch {epoch}/{cfg.train_iter_egn}, obj={obj_sum / len(train_dataset)}')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}.')


####################################
#            testing               #
####################################

if cfg.test_data_type == 'random':
    dataset = get_random_data(cfg.num_data, cfg.dim, 1, device)
elif cfg.test_data_type == 'starbucks':
    dataset = get_starbucks_data(device)
else:
    raise ValueError(f'Unknown dataset name {cfg.train_dataset_type}!')

for index, (prob_name, points) in enumerate(dataset):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(points[:, 0].cpu(), points[:, 1].cpu(), 'b.')
    method_idx = 0
    print('-' * 20)
    print(f'{prob_name} points={len(points)} select={cfg.test_num_centers}')
    ws.write(index + 1, 0, prob_name)

    if 'greedy' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        cluster_centers, selected_indices = greedy_facility_location(points, cfg.test_num_centers, distance=cfg.distance_metric, device=points.device)
        objective = compute_objective(points, cluster_centers, cfg.distance_metric).item()
        print(f'{prob_name} greedy objective={objective:.4f} '
              f'selected={sorted(selected_indices.cpu().numpy().tolist())} '
              f'time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'greedy-objective')
            ws.write(0, method_idx * 2, 'greedy-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    if 'gurobi' in cfg.methods:
        # Gurobi - integer programming
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = gurobi_facility_location(
            points, cfg.test_num_centers, distance=cfg.distance_metric, linear_relaxation=False,
            timeout_sec=cfg.solver_timeout, verbose=cfg.verbose)
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        print(f'{prob_name} Gurobi objective={ip_obj:.4f} '
              f'selected={sorted(top_k_indices.cpu().numpy().tolist())} '
              f'time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'Gurobi-objective')
            ws.write(0, method_idx * 2, 'Gurobi-time')
        ws.write(index + 1, method_idx * 2 - 1, ip_obj)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    if 'scip' in cfg.methods:
        # SCIP - integer programming
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = ortools_facility_location(
            points, cfg.test_num_centers, distance=cfg.distance_metric, linear_relaxation=False,
            timeout_sec=cfg.solver_timeout, solver_name='SCIP')
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        print(f'{prob_name} SCIP objective={ip_obj:.4f} '
              f'selected={sorted(top_k_indices.cpu().numpy().tolist())} '
              f'time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'SCIP-objective')
            ws.write(0, method_idx * 2, 'SCIP-time')
        ws.write(index + 1, method_idx * 2 - 1, ip_obj)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    if 'egn' in cfg.methods:
        # Erdos Goes Neural
        method_idx += 1
        model.load_state_dict(torch.load(f'facility_location_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_egn.pt'))
        objective, selected_indices, finish_time = egn_facility_location(
            points, cfg.test_num_centers, model, cfg.softmax_temp, cfg.egn_beta,
            time_limit=-1, distance_metric=cfg.distance_metric)
        print(f'{prob_name} EGN objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'EGN-objective')
            ws.write(0, method_idx * 2, 'EGN-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

        method_idx += 1
        objective, selected_indices, finish_time = egn_facility_location(
            points, cfg.test_num_centers, model, cfg.softmax_temp, cfg.egn_beta, cfg.egn_trials,
            time_limit=-1, distance_metric=cfg.distance_metric)
        print(f'{prob_name} EGN-accu objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'EGN-accu-objective')
            ws.write(0, method_idx * 2, 'EGN-accu-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    if 'cardnn-s' in cfg.methods:
        # CardNN-S
        method_idx += 1
        model.load_state_dict(torch.load(f'facility_location_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_cardnn-s.pt'))
        objective, selected_indices, finish_time = sinkhorn_facility_location(
            points, cfg.test_num_centers, model,
            cfg.softmax_temp, 1, 0, cfg.sinkhorn_tau, cfg.sinkhorn_iter, cfg.soft_opt_iter,
            time_limit=-1, verbose=cfg.verbose, distance_metric=cfg.distance_metric)
        print(f'{prob_name} CardNN-S objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-S-objective')
            ws.write(0, method_idx * 2, 'CardNN-S-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    if 'cardnn-gs' in cfg.methods:
        # CardNN-GS
        method_idx += 1
        model.load_state_dict(torch.load(f'facility_location_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_cardnn-gs.pt'))
        objective, selected_indices, finish_time = sinkhorn_facility_location(
            points, cfg.test_num_centers, model,
            cfg.softmax_temp, cfg.gumbel_sample_num, cfg.gumbel_sigma, cfg.sinkhorn_tau, cfg.sinkhorn_iter, cfg.gs_opt_iter,
            time_limit=-1, verbose=cfg.verbose, distance_metric=cfg.distance_metric)

        print(f'{prob_name} CardNN-GS objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-GS-objective')
            ws.write(0, method_idx * 2, 'CardNN-GS-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    if 'cardnn-hgs' in cfg.methods:
        # CardNN-HGS
        method_idx += 1
        model.load_state_dict(torch.load(f'facility_location_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_cardnn-gs.pt'))
        objective, selected_indices, finish_time = sinkhorn_facility_location(
            points, cfg.test_num_centers, model,
            cfg.softmax_temp, cfg.gumbel_sample_num, cfg.homotophy_sigma, cfg.homotophy_tau, cfg.homotophy_sk_iter, cfg.homotophy_opt_iter,
            time_limit=-1, verbose=cfg.verbose, distance_metric=cfg.distance_metric)
        print(f'{prob_name} CardNN-HGS objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-HGS-objective')
            ws.write(0, method_idx * 2, 'CardNN-HGS-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    wb.save(f'facility_location_result_{cfg.test_data_type}_{cfg.test_num_centers}-{cfg.num_data}_{timestamp}.xls')
    plt.close()
