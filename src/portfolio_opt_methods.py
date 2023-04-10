import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from src.gumbel_sinkhorn_topk import gumbel_sinkhorn_topk


#################################################
#              Helper Functions                 #
#################################################

def compute_measures(returns, annual_factor=252):
    """
    Compute mu (expected return) and cov (risk)
    :param returns: (NxF) daily returns w.r.t. to the previous day. N: number of days, F: number of assets
    :param annual_factor: compute annual return from daily return
    :return: mu, cov
    """
    assert len(returns.shape) == 2
    mu = torch.mean(returns, dim=0)
    returns_minus_mu = returns - mu.unsqueeze(0)
    cov = torch.mm(returns_minus_mu.t(), returns_minus_mu) / (returns.shape[0] - 1)
    return mu * annual_factor, cov * annual_factor


def compute_sharpe_ratio(mu, cov, weight, rf, return_details=False):
    """
    Compute the Sharpe ratio of the given portfolio
    :param mu: (F) expected return. F: number of assets
    :param cov: (FxF) risk matrix
    :param weight: (F) or (BxF) weight of assets of the portfolio. B: the batch size
    :param rf: risk-free return
    :return: the Sharpe ratio
    """
    if len(weight.shape) == 1:
        weight = weight.unsqueeze(0)
        batched_input = False
    else:
        batched_input = True
    assert len(weight.shape) == 2
    mu = mu.unsqueeze(0)
    cov = cov.unsqueeze(0)
    returns = (mu * weight).sum(dim=1)
    risk = torch.sqrt(torch.matmul(torch.matmul(weight.unsqueeze(1), cov), weight.unsqueeze(2))).squeeze()
    sharpe = (returns - rf) / risk
    if not batched_input:
        sharpe = sharpe.squeeze()
        risk = risk.squeeze()
        returns = returns.squeeze()
    if return_details:
        return sharpe, risk, returns
    else:
        return sharpe


def is_pos_def(A):
    """
    Check if a matrix is positive definite
    """
    A = A.detach().cpu().numpy()
    if np.sum(np.abs(A - A.T)) < 1e-8:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError as err:
            if 'Matrix is not positive definite' in err.args:
                return False
            else:
                raise
    else:
        return False


def is_psd(A, tol=1e-8):
    """
    Check if a matrix is positive semi-definite
    """
    A = A.detach().cpu().numpy()
    if np.sum(np.abs(A - A.T)) < 1e-8:
        E = np.linalg.eigvalsh(A)
        return np.all(E > -tol)
    else:
        return False


def torch_sqrtm(m):
    """
    Square root of a matrix
    inspired by: https://discuss.pytorch.org/t/raising-a-tensor-to-a-fractional-power/93655/3
    """
    assert is_psd(m), 'Input matrix is not positive semi-definitive'

    evals, evecs = torch.eig(m, eigenvectors=True)  # get eigendecomposition
    evals = torch.relu(evals[:, 0])  # get real part of (real) eigenvalues
    evpow = evals ** (1 / 2)  # raise eigenvalues to fractional power

    # build exponentiated matrix from exponentiated eigenvalues
    mpow = torch.matmul(evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs)))
    return mpow


def sqrt_newton_schulz_autograd(A, numIters=10):
    """
    Square root of a matrix by Newton-Schulz iterations
    Copied from: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    """
    if len(A.shape) == 2:
        batched_input = False
        A = A.unsqueeze(0)
    else:
        batched_input = True
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).to(dtype=A.dtype, device=A.device)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).to(dtype=A.dtype, device=A.device)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)

    if batched_input:
        return sA
    else:
        return sA.squeeze(0)


#################################################
#         Learning/Prediction Methods           #
#################################################

class LSTMModel(torch.nn.Module):
    # Encoder-decoder LSTM model
    def __init__(self, num_assets, feature_size, num_layers=3, inp_scale=50, outp_scale=0.1):
        super(LSTMModel, self).__init__()
        self.encoder = torch.nn.LSTM(input_size=num_assets, hidden_size=feature_size, num_layers=num_layers)
        self.decoder = torch.nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers)
        self.topk_fc = torch.nn.Linear(num_layers * feature_size * 4, num_assets)
        self.price_fc = torch.nn.Linear(feature_size, num_assets)
        self.inp_scale = inp_scale
        self.outp_scale = outp_scale
        self.num_assets = num_assets
        self.opt_layer = None # created upon needed

    def get_opt_layer(self):
        """
        Get a CVXPY differentiable optimization layer
        """
        varW = cp.Variable(self.num_assets)
        paramTopK = cp.Parameter(self.num_assets)
        paramMu = cp.Parameter(self.num_assets)
        paramSqrtCov = cp.Parameter((self.num_assets, self.num_assets))
        rf = cp.Parameter()
        obj = cp.Minimize(cp.sum_squares(varW @ paramSqrtCov))
        cons = [varW <= paramTopK * sum(varW), varW >= 0, varW.T @ (paramMu - rf) == 1]
        prob = cp.Problem(obj, cons)
        opt_layer = CvxpyLayer(prob, parameters=[paramTopK, paramMu, paramSqrtCov, rf], variables=[varW])
        return opt_layer

    def forward(self, seq, pred_len, mode='pred', rf=0.03, k=5,
                gumbel_sample_num=1000, gumbel_noise_fact=0.1, return_best_weight=False):
        pred_seq, top_k_select = self.network_forward(seq * self.inp_scale, pred_len)
        pred_seq = pred_seq * self.outp_scale
        if mode == 'pred': # only predict the future price
            return pred_seq
        elif mode == 'predict-then-opt': # predict-then-optimize
            mu, cov = compute_measures(pred_seq)
            _, weight, __ = gurobi_portfolio_opt(mu, cov, rf=rf, obj='Sharpe', card_constr=k,
                                                 linear_relaxation=False)
            if weight is None: # solver error, return a trivial solution
                weight = torch.ones_like(mu) / mu.shape[0]
            return pred_seq, weight
        elif mode == 'history-opt':
            mu, cov = compute_measures(seq)
            _, weight, __ = gurobi_portfolio_opt(mu, cov, rf=rf, obj='Sharpe', card_constr=k,
                                                 linear_relaxation=False)
            if weight is None: # solver error, return a trivial solution
                weight = torch.ones_like(mu) / mu.shape[0]
            return pred_seq, weight
        elif mode == 'predict-and-opt': # joint predict and optimize by Gumbel Sinkhorn TopK
            mu, cov = compute_measures(pred_seq)
            if k > 0:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    top_k_select, k, max_iter=100, tau=.05,
                    sample_num=gumbel_sample_num, noise_fact=gumbel_noise_fact, return_prob=True
                )
            else:
                probs = torch.ones_like(mu)
            z = torch.mm(torch.inverse(cov), (mu.unsqueeze(1) - rf)).squeeze(1)
            z = z.unsqueeze(0) * probs
            weight = torch.relu(z)
            weight = weight / weight.sum(dim=-1, keepdim=True)
            if return_best_weight:
                sharpe = compute_sharpe_ratio(mu, cov, weight, rf)
                weight = weight[torch.argmax(sharpe)]
            return pred_seq, weight
        else:
            raise ValueError(f'mode={mode} is not supported.')

    def network_forward(self, seq, pred_len):
        """
        :param seq: (NxBxF) or (NxF). N: length of time sequence. B: batch size. F: feature dimension
        :param pred_len: length of predicted sequence
        :return: predicted sequence (NxBxF) or (NxF), predicted subset selection
        """
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(1)  # auxiliary batch dimension
            batch_size = 1
            batched_input = False
        else:
            batch_size = seq.shape[1]
            batched_input = True

        latent_feat, encoder_hidden = self.encoder(seq)
        latent_feat = latent_feat[-1:, :, :]
        decoder_hidden = None
        pred_seq = []
        for i in range(pred_len):
            pred_elem, decoder_hidden = self.decoder(latent_feat, decoder_hidden)
            pred_seq.append(pred_elem)
        selection = self.topk_fc(torch.cat(encoder_hidden + decoder_hidden, dim=-1).permute(1, 0, 2).reshape(batch_size, -1))
        selection = torch.sigmoid(selection)
        pred_seq = self.price_fc(torch.cat(pred_seq, dim=0))
        pred_seq = pred_seq - torch.tanh(pred_seq)  # TanhShrink activation

        if not batched_input:
            pred_seq = pred_seq.squeeze(1)
            selection = selection.squeeze(0)
        return pred_seq, selection


#################################################
#          Optimization-based Methods           #
#################################################

def gurobi_portfolio_opt(
        mu: torch.Tensor, # return of each asset
        cov: torch.Tensor, # covariance
        r: float=None, # expected return
        rf: float=0, # risk-free return
        obj='MinRisk', # objective function, default is minimize risk given expected return.
        card_constr: int=-1,
        linear_relaxation=True,
        timeout_sec=60,
        start=None,
        log_to_console=False
):
    import gurobipy as grb

    try:
        model = grb.Model('portfolio optimization')
        if not log_to_console:
            model.setParam('LogToConsole', 0)
        if timeout_sec > 0:
            model.setParam('TimeLimit', timeout_sec)

        device = mu.device

        mu = mu.cpu()
        cov = cov.cpu()

        # Initialize variables
        VarX = {}
        VarW = {}
        for i in range(mu.shape[0]):
            if linear_relaxation:
                VarX[i] = model.addVar(0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f'x_{i}')
            else:
                VarX[i] = model.addVar(vtype=grb.GRB.BINARY, name=f'x_{i}')
            if start is not None:
                VarX[i].start = start[i]
            VarW[i] = model.addVar(lb=0.0, vtype=grb.GRB.CONTINUOUS, name=f'w_{i}')

        # Weight
        weight_sum = 0
        for i in range(mu.shape[0]):
            weight_sum += VarW[i]

        # Cardinality Constraint
        if card_constr > 0:
            X_constr = 0
            for i in range(mu.shape[0]):
                X_constr += VarX[i]
            model.addConstr(X_constr <= card_constr)

            for i in range(mu.shape[0]):
                if obj == 'Sharpe':
                    model.addConstr(VarW[i] <= VarX[i] * weight_sum)
                else:
                    model.addConstr(VarW[i] <= VarX[i])

        # Return
        return_sum = 0
        for i in range(mu.shape[0]):
            return_sum += VarW[i] * (mu[i] - rf)

        # The risk
        risk = 0
        for i in range(mu.shape[0]):
            for j in range(mu.shape[0]):
                risk += VarW[i] * VarW[j] * cov[i, j]
        risk = risk

        # Consider two different objective functions ('MinRisk' or 'Sharpe')
        if obj == 'MinRisk':
            assert r is not None
            model.addConstr(weight_sum == 1) # add weight constraint
            model.addConstr(return_sum == r) # add return value constraint
            model.setObjective(risk, grb.GRB.MINIMIZE)
        elif obj == 'Sharpe':
            # Handling Sharpe objective needs a special trick, VarW is treated as auxiliary variables
            # see https://support.gurobi.com/hc/en-us/community/posts/360074491212-Divisor-must-be-a-constant
            model.addConstr(return_sum == 1)
            model.setObjective(risk, grb.GRB.MINIMIZE)
        else:
            raise ValueError('Unknown objective function:', obj)

        model.optimize()

        if not hasattr(model.getVarByName('x_0'), 'X'):
            return None, None, None  # no solution

        res = [model.getVarByName(f'x_{i}').X for i in range(mu.shape[0])]
        if linear_relaxation:
            res = np.array(res, dtype=np.float)
        else:
            res = np.array(res, dtype=np.int)
        weight = np.array([model.getVarByName(f'w_{i}').X for i in range(mu.shape[0])])
        if obj == 'Sharpe':
            weight = weight / weight.sum()
        return model.getObjective().getValue(), torch.from_numpy(weight).to(device), torch.from_numpy(res).to(device)

    except grb.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
