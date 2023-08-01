from src.max_covering_methods import *
import time
import xlwt
from datetime import datetime
import os
from src.config import load_config
from src.max_covering_data import get_random_dataset, get_twitch_dataset

####################################
#             config               #
####################################

cfg = load_config()
device = torch.device('cuda:0')


####################################
#            training              #
####################################

if cfg.train_data_type == 'random':
    train_dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 1)
elif cfg.train_data_type == 'twitch':
    train_dataset = get_twitch_dataset()
else:
    raise ValueError(f'Unknown training dataset {cfg.train_data_type}!')

model = GNNModel().to(device)

for method_name in cfg.methods:
    model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_{method_name}.pt'
    if not os.path.exists(model_path) and method_name in ['cardnn-gs', 'cardnn-s', 'egn']:
        print(f'Training the model weights for {method_name}...')
        model = GNNModel().to(device)
        if method_name in ['cardnn-gs', 'cardnn-s']:
            train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
            for epoch in range(cfg.train_iter):
                # training loop
                obj_sum = 0
                for name, weights, sets in train_dataset:
                    bipartite_adj = None
                    graph = build_graph_from_weights_sets(weights, sets, device)
                    latent_vars = model(graph)
                    if method_name == 'cardnn-gs':
                        sample_num = cfg.train_gumbel_sample_num
                        noise_fact = cfg.gumbel_sigma
                    else:
                        sample_num = 1
                        noise_fact = 0
                    top_k_indices, probs = gumbel_sinkhorn_topk(
                        latent_vars, cfg.train_max_covering_items, max_iter=cfg.sinkhorn_iter, tau=cfg.sinkhorn_tau,
                        sample_num=sample_num, noise_fact=noise_fact, return_prob=True
                    )
                    # compute objective by softmax
                    obj, _ = compute_obj_differentiable(weights, sets, probs, bipartite_adj, device=probs.device)
                    (-obj).mean().backward()
                    obj_sum += obj.mean()

                    train_optimizer.step()
                    train_optimizer.zero_grad()

                print(f'epoch {epoch}/{cfg.train_iter}, obj={obj_sum / len(train_dataset)}')
        if method_name in ['egn']:
            train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
            # training loop
            for epoch in range(cfg.train_iter):
                obj_sum = 0
                for name, weights, sets in train_dataset:
                    bipartite_adj = None
                    graph = build_graph_from_weights_sets(weights, sets, device)
                    probs = model(graph)
                    constraint_conflict = torch.relu(probs.sum() - cfg.train_max_covering_items)
                    obj, _ = compute_obj_differentiable(weights, sets, probs, bipartite_adj, device=probs.device)
                    obj = obj - cfg.egn_beta * constraint_conflict
                    (-obj).mean().backward()
                    obj_sum += obj.mean()

                    train_optimizer.step()
                    train_optimizer.zero_grad()
                print(f'epoch {epoch}/{cfg.train_iter}, obj={obj_sum / len(train_dataset)}')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}.')


####################################
#            testing               #
####################################

if cfg.test_data_type == 'random':
    dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 0)
elif cfg.test_data_type == 'twitch':
    dataset = get_twitch_dataset()
else:
    raise ValueError(f'Unknown testing dataset {cfg.test_data_type}!')

wb = xlwt.Workbook()
ws = wb.add_sheet(f'max_covering_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}')
ws.write(0, 0, 'name')
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

torch.random.manual_seed(1)
for index, (name, weights, sets) in enumerate(dataset):
    method_idx = 0
    print('-' * 20)
    print(f'{name} items={len(weights)} sets={len(sets)}')
    ws.write(index + 1, 0, name)

    # greedy
    if 'greedy' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        objective, selected = greedy_max_covering(weights, sets, cfg.test_max_covering_items)
        print(f'{name} greedy objective={objective} selected={sorted(selected)} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'greedy-objective')
            ws.write(0, method_idx * 2, 'greedy-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # SCIP - integer programming
    if 'scip' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = ortools_max_covering(weights, sets, cfg.test_max_covering_items, solver_name='SCIP', linear_relaxation=False, timeout_sec=cfg.solver_timeout)
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        top_k_indices = sorted(top_k_indices.cpu().numpy().tolist())
        print(f'{name} SCIP objective={ip_obj:.0f} selected={top_k_indices} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'SCIP-objective')
            ws.write(0, method_idx * 2, 'SCIP-time')
        ws.write(index + 1, method_idx * 2 - 1, ip_obj)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # Gurobi - integer programming
    if 'gurobi' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        ip_obj, ip_scores = gurobi_max_covering(weights, sets, cfg.test_max_covering_items, linear_relaxation=False, timeout_sec=cfg.solver_timeout, verbose=cfg.verbose)
        ip_scores = torch.tensor(ip_scores)
        top_k_indices = torch.nonzero(ip_scores, as_tuple=False).view(-1)
        top_k_indices = sorted(top_k_indices.cpu().numpy().tolist())
        print(f'{name} Gurobi objective={ip_obj:.0f} selected={top_k_indices} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'Gurobi-objective')
            ws.write(0, method_idx * 2, 'Gurobi-time')
        ws.write(index + 1, method_idx * 2 - 1, ip_obj)
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    # Erdos Goes Neural
    if 'egn' in cfg.methods:
        method_idx += 1
        model.load_state_dict(torch.load(f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_egn.pt'))
        objective, best_top_k_indices, finish_time = egn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.egn_beta)
        print(f'{index} EGN objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'EGN-objective')
            ws.write(0, method_idx * 2, 'EGN-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

        method_idx += 1
        objective, best_top_k_indices, finish_time = egn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.egn_beta, cfg.egn_trials)
        print(f'{index} EGN-accu objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={finish_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'EGN-accu-objective')
            ws.write(0, method_idx * 2, 'EGN-accu-time')
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)

    # CardNN-S
    if 'cardnn-s' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-s.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = cardnn_max_covering(weights, sets, cfg.test_max_covering_items, model, 1, 0,
                                                           cfg.sinkhorn_tau, cfg.sinkhorn_iter, cfg.soft_opt_iter,
                                                           verbose=cfg.verbose)
        print(f'{name} CardNN-S objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-S-objective')
            ws.write(0, method_idx * 2, 'CardNN-S-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # CardNN-GS
    if 'cardnn-gs' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-gs.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = cardnn_max_covering(weights, sets, cfg.test_max_covering_items, model,
                                                           cfg.gumbel_sample_num, cfg.gumbel_sigma, cfg.sinkhorn_tau,
                                                           cfg.sinkhorn_iter, cfg.gs_opt_iter, verbose=cfg.verbose)
        print(f'{name} CardNN-GS objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-GS-objective')
            ws.write(0, method_idx * 2, 'CardNN-GS-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # CardNN-HGS
    if 'cardnn-hgs' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-gs.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = cardnn_max_covering(weights, sets, cfg.test_max_covering_items, model,
                                                           cfg.gumbel_sample_num, cfg.homotophy_sigma,
                                                           cfg.homotophy_tau, cfg.homotophy_sk_iter,
                                                           cfg.homotophy_opt_iter, verbose=cfg.verbose)
        print(f'{name} CardNN-HGS objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'CardNN-HGS-objective')
            ws.write(0, method_idx * 2, 'CardNN-HGS-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # perturb-TopK
    if 'perturb-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-gs.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = gumbel_max_covering(weights, sets, cfg.test_max_covering_items, model,
                                                           cfg.gumbel_sample_num * 10, # needs 10x more samples than others
                                                           cfg.perturb_sigma, cfg.perturb_opt_iter, verbose=cfg.verbose)
        print(f'{name} perturb-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'perturb-TopK-objective')
            ws.write(0, method_idx * 2, 'perturb-TopK-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # blackbox-TopK
    if 'blackbox-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-gs.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = blackbox_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.blackbox_lambda, cfg.blackbox_opt_iter, verbose=cfg.verbose)
        print(f'{name} blackbox-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'blackbox-TopK-objective')
            ws.write(0, method_idx * 2, 'blackbox-TopK-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    # LML-TopK
    if 'lml-topk' in cfg.methods:
        method_idx += 1
        prev_time = time.time()
        model_path = f'max_covering_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_cardnn-gs.pt'
        model.load_state_dict(torch.load(model_path))
        best_obj, best_top_k_indices = lml_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.lml_opt_iter, verbose=cfg.verbose)
        print(
            f'{name} LML-TopK objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
        if index == 0:
            ws.write(0, method_idx * 2 - 1, 'LML-TopK-objective')
            ws.write(0, method_idx * 2, 'LML-TopK-time')
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
        ws.write(index + 1, method_idx * 2, time.time() - prev_time)

    wb.save(f'max_covering_result_{cfg.test_data_type}_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_{timestamp}.xls')
