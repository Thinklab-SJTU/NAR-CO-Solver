import matplotlib
matplotlib.use('Agg')

from torch.utils.tensorboard import SummaryWriter
from src.portfolio_opt_methods import *
from src.portfolio_opt_data import PortDataset


######################################
#           Hyperparameters          #
######################################

START_EPOCH = 0
NUM_EPOCHS = 10000
HISTORY_LEN = 120
FUTURE_LEN = 120
NUM_FEATURE = 32
DEVICE = torch.device('cuda:0')
LR = 1e-3
LR_STEPS = [1500]
RF = 0.03 # risk-free return to compute the Sharpe ratio
K = 20 # for the cardinality (topK) constraint


######################################
#              training              #
######################################

def train_test_portfolio(model, train_set, mse_weight=1, sharpe_weight=1, opt_method='predict-and-opt', test_set=None, test_items='all'):
    assert mse_weight > 0 or sharpe_weight > 0

    working_lr_steps = []
    for step in LR_STEPS:
        if step - START_EPOCH < 0:
            continue
        working_lr_steps.append(step - START_EPOCH)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, working_lr_steps)

    writer = SummaryWriter()
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        train_num = len(train_set)
        epoch_mse = 0
        epoch_sharpe = 0
        for iter_idx, train_data in enumerate(np.random.permutation(train_set)):
            # prepare data
            history = torch.tensor(train_data['history'].values).to(device=DEVICE, dtype=torch.double)
            future = torch.tensor(train_data['future'].values).to(device=DEVICE, dtype=torch.double)

            # forward pass
            if sharpe_weight > 0:
                pred_seq, weight = model(history, FUTURE_LEN, opt_method, RF, K,
                                         gumbel_sample_num=10 if opt_method == 'jpo-old' else 1000)
            else:
                pred_seq = model(history, FUTURE_LEN)

            # compute loss
            loss = 0
            mse = torch.sum((pred_seq - future) ** 2) / FUTURE_LEN
            writer.add_scalar('train_mse', mse.detach(), iter_idx + epoch * train_num)
            epoch_mse += mse.detach() / train_num
            if mse_weight > 0:
                loss += mse_weight * mse
            if sharpe_weight > 0:
                mu, cov = compute_measures(future)
                sharpe = compute_sharpe_ratio(mu, cov, weight, RF)
                writer.add_scalars(
                    'train_sharpe',
                    {'mean': sharpe.mean().detach(), 'min': sharpe.min().detach(), 'max': sharpe.max().detach()},
                    iter_idx + epoch * train_num
                )
                loss += - sharpe_weight * sharpe.mean()
                epoch_sharpe += sharpe.mean().detach() / train_num
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()
        writer.add_scalar('epoch_mse', epoch_mse, (epoch + 1) * train_num)
        if sharpe_weight > 0:
            writer.add_scalar('epoch_sharpe', epoch_sharpe, (epoch + 1) * train_num)
        print(f'e={epoch}/{NUM_EPOCHS}, loss={mse_weight * epoch_mse - sharpe_weight * epoch_sharpe:.6f}')

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f'output/portfolio_lstm_epoch{epoch}_{opt_method}.pt')
        if epoch % 1 == 0 and test_set is not None:
            test_result = test_portfolio(model, test_set, test_items, verbose=True)
            writer.add_scalars('test',
                               {k: sum(v) / len(v) for k, v in test_result.items()},
                               (epoch + 1) * train_num)


######################################
#              testing               #
######################################

def test_portfolio(model, test_set, test_items='all', verbose=True):
    with torch.set_grad_enabled(False):
        if test_items == 'all':
            test_items = ['mse', 'predict-and-opt', 'history-opt', 'predict-then-opt']
        return_dict = {k + '-return': [] for k in set(test_items)-set(['mse'])}
        return_dict.update({k + '-risk' : [] for k in set(test_items)-set(['mse'])})
        if 'mse' in test_items:
            return_dict.update({'mse' : []})
        

        for iter_idx, test_data in enumerate(test_set):
            history = torch.tensor(test_data['history'].values).to(device=DEVICE, dtype=torch.double)
            future = torch.tensor(test_data['future'].values).to(device=DEVICE, dtype=torch.double)
            mu, cov = compute_measures(future)

            test_print = [f'test id={iter_idx}, date={test_data["real_date"]}']

            # simple price prediction
            if 'mse' in test_items:
                pred_seq = model(history, FUTURE_LEN)
                mse = torch.sum((pred_seq - future) ** 2) / FUTURE_LEN
                test_print.append(f'pred_mse={mse:.4f}')
                return_dict['mse'].append(mse)

            # cardinality constrained predict-and-optimize
            if 'predict-and-opt' in test_items:
                _, jpo_weight = model(history, FUTURE_LEN, 'predict-and-opt', RF, K, gumbel_sample_num=1000, gumbel_noise_fact=0.1, return_best_weight=True)
                sharpe, risk, return_ = compute_sharpe_ratio(mu, cov, jpo_weight, RF, return_details=True)
                test_print.append(f'predict-and-opt: sharpe={sharpe:.4f}, return={return_:.4f}, risk={risk:.4f}')
                return_dict['predict-and-opt-return'].append(return_)
                return_dict['predict-and-opt-risk'].append(risk)

            # find and return the best portfolio in history
            if 'history-opt' in test_items:
                _, history_weight = model(history, FUTURE_LEN, 'history-opt', RF, K)
                sharpe, risk, return_ = compute_sharpe_ratio(mu, cov, history_weight, RF, return_details=True)
                test_print.append(f'history-opt: sharpe={sharpe:.4f}, return={return_:.4f}, risk={risk:.4f}')
                return_dict['history-opt-return'].append(return_.detach())
                return_dict['history-opt-risk'].append(risk.detach())

            # cardinality constrained predict-then-optimize
            if 'predict-then-opt' in test_items:
                _, pto_weight = model(history, FUTURE_LEN, 'predict-then-opt', RF, K)
                sharpe, risk, return_ = compute_sharpe_ratio(mu, cov, pto_weight, RF, return_details=True)
                test_print.append(f'predict-then-opt: sharpe={sharpe:.4f}, return={return_:.4f}, risk={risk:.4f}')
                return_dict['predict-then-opt-return'].append(return_)
                return_dict['predict-then-opt-risk'].append(risk)

            if verbose:
                print(', '.join(test_print))
        print('Evaluation complete.')
        for k, v in return_dict.items():
            print(f'{k}: {sum(v) / len(v):.4f}')
            if 'risk' in k: # print the sharpe ratio
                r_k = '-'.join(k.split('-')[:-1])
                r_v = return_dict[r_k + '-return']
                print(r_k)
                print(r_v)
                print(f'{r_k + "-sharpe"}: {(sum(r_v) / len(r_v) - RF) / (sum(v) / len(v)):.4f}')

    return return_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CardNN experiment protocol (predictive portfolio optimization).')
    parser.add_argument("--train", help="enable training", action="store_true")
    parser.add_argument('--method', help="the name of method "
                                         "(from \'predict-and-opt\', \'predict-then-opt\', \'history-opt\')",
                        default='predict-and-opt', type=str)
    parser.add_argument('--start_epoch', help="the number of starting epoch", default=0, type=int)
    args = parser.parse_args()

    dataset = PortDataset('snp500', HISTORY_LEN, FUTURE_LEN, train_test_split=0.75)
    model = LSTMModel(dataset.num_assets, NUM_FEATURE, 1).to(device=DEVICE)
    model.double()

    if args.start_epoch > 0:
        pretrained_path = f'output/portfolio_lstm_epoch{args.start_epoch}_{args.method}.pt'
        print(f'Loading model weights from {pretrained_path}...')
        model.load_state_dict(torch.load(pretrained_path))

    if args.train:
        train_test_portfolio(model, dataset.train_set, mse_weight=0, sharpe_weight=1, opt_method=args.method,
                             test_set=dataset.test_set, test_items=['mse', args.method])
    else:
       test_portfolio(model, dataset.test_set, test_items=['mse', args.method], verbose=True)
