import torch
from ortools.linear_solver import pywraplp
import torch.nn as nn
import warnings


def ortools_ot(cost, row_marginal, col_marginal, solver_name='scip'):
    if len(cost.shape) == 3:
        batched_input = True
    else:
        batched_input = False
        cost = cost.unsqueeze(0)
        row_marginal = row_marginal.unsqueeze(0)
        col_marginal = col_marginal.unsqueeze(0)

    device = cost.device

    batch_size = cost.shape[0]
    output = []
    for b in range(batch_size):
        solver = pywraplp.Solver.CreateSolver(solver_name)
        VarX = {}
        objective = 0
        for i in range(row_marginal.shape[1]):
            for j in range(col_marginal.shape[1]):
                VarX[f'{i}-{j}'] = solver.NumVar(lb=0, ub=100, name=f'x_{i}-{j}')
                objective += VarX[f'{i}-{j}'] * cost[b, i, j].item()

        for i in range(row_marginal.shape[1]):
            constraint_i = 0
            for j in range(col_marginal.shape[1]):
                constraint_i += VarX[f'{i}-{j}']
            solver.Add(constraint_i - row_marginal[b, i].item() <= 1e-5)
            solver.Add(constraint_i - row_marginal[b, i].item() >= -1e-5)

        for j in range(col_marginal.shape[1]):
            constraint_j = 0
            for i in range(row_marginal.shape[1]):
                constraint_j += VarX[f'{i}-{j}']
            solver.Add(constraint_j - col_marginal[b, j].item() <= 1e-5)
            solver.Add(constraint_j - col_marginal[b, j].item() >= -1e-5)

        solver.Minimize(objective)
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError('OT solver not optimal')
        output.append(solver.Objective().Value())

        solved_x = torch.zeros(row_marginal.shape[1], col_marginal.shape[1])
        for i in range(row_marginal.shape[1]):
            for j in range(col_marginal.shape[1]):
                solved_x[i, j] = VarX[f'{i}-{j}'].solution_value()

    if batched_input:
        return torch.tensor(output, device=device)
    else:
        return output[0]


# The following "Declarative" Sinkhorn functions are modified from
# https://github.com/anucvml/ddn/blob/master/ddn/pytorch/optimal_transport.py

def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(B, 1, W) * c

    return P


def _sinkhorn_inline(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """As above but with inline calculations for when autograd is not needed."""

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
        raise NotImplementedError
    else:
        # P = torch.exp(-1.0 * gamma * (M - torch.min(M, 2, keepdim=True).values))
        log_P = -1.0 * gamma * (M - torch.min(M, 2, keepdim=True).values)
        log_r = torch.log(r)
        log_c = torch.log(c)

    for i in range(maxiters):
        log_alpha = torch.logsumexp(log_P, 2) # alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        log_P -= log_alpha.view(B, H, 1) # P /= alpha.view(B, H, 1)
        log_P += log_r # P *= r

        log_beta = torch.logsumexp(log_P, 1) # beta = torch.sum(P, 1)
        if torch.max(torch.abs(torch.exp(log_beta) - c)) <= eps:
            break
        log_P -= log_beta.view(B, 1, W) # P /= beta.view(B, 1, W)
        log_P += log_c #P *= c

    return torch.exp(log_P)


class OptimalTransportFcn(torch.autograd.Function):
    """
    PyTorch autograd function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector

    Allows for approximate gradient calculations, which is faster and may be useful during early stages of learning,
    when exp(-\gamma M) is already nearly doubly stochastic, or when gradients are otherwise noisy.

    Both r and c must be positive, if provided. They will be normalized to sum to one.
    """

    @staticmethod
    def forward(ctx, M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        """Solve optimal transport using skinhorn. Method can be 'block', 'full', 'fullchol' or 'approx'."""
        assert method in ('block', 'full', 'fullchol', 'approx')

        with torch.no_grad():
            # normalize r and c to ensure that they sum to one (and save normalization factor for backward pass)
            if r is not None:
                ctx.inv_r_sum = 1.0 / torch.sum(r, dim=1, keepdim=True)
                r = ctx.inv_r_sum * r
            if c is not None:
                ctx.inv_c_sum = 1.0 / torch.sum(c, dim=1, keepdim=True)
                c = ctx.inv_c_sum * c

            # run sinkhorn
            P = _sinkhorn_inline(M, r, c, gamma, eps, maxiters, logspace)

        ctx.save_for_backward(M, r, c, P)
        ctx.gamma = gamma
        ctx.logspace = logspace
        ctx.method = method

        return P

    @staticmethod
    def backward(ctx, dJdP):
        """Implement backward pass using implicit differentiation."""

        M, r, c, P = ctx.saved_tensors
        B, H, W = M.shape

        # initialize backward gradients (-v^T H^{-1} B with v = dJdP and B = I or B = -1/r or B = -1/c)
        dJdM = -1.0 * ctx.gamma * P * dJdP
        dJdr = None if not ctx.needs_input_grad[1] else torch.zeros_like(r)
        dJdc = None if not ctx.needs_input_grad[2] else torch.zeros_like(c)

        # return approximate gradients
        if ctx.method == 'approx':
            return dJdM, dJdr, dJdc, None, None, None, None, None, None

        # compute exact row and column sums (in case of small numerical errors or forward pass not converging)
        alpha = torch.sum(P, dim=2)
        beta = torch.sum(P, dim=1)

        # compute [vHAt1, vHAt2] = v^T H^{-1} A^T as two blocks
        vHAt1 = torch.sum(dJdM[:, 1:H, 0:W], dim=2).view(B, H - 1, 1)
        vHAt2 = torch.sum(dJdM, dim=1).view(B, W, 1)

        # compute [v1, v2] = -v^T H^{-1} A^T (A H^{-1] A^T)^{-1}
        if ctx.method == 'block':
            # by block inverse of (A H^{-1] A^T)
            PdivC = P[:, 1:H, 0:W] / beta.view(B, 1, W)
            RminusPPdivC = torch.diag_embed(alpha[:, 1:H]) - torch.bmm(P[:, 1:H, 0:W], PdivC.transpose(1, 2))
            try:
                block_11 = torch.linalg.cholesky(RminusPPdivC)
            except:
                # block_11 = torch.ones((B, H-1, H-1), device=M.device, dtype=M.dtype)
                block_11 = torch.eye(H - 1, device=M.device, dtype=M.dtype).view(1, H - 1, H - 1).repeat(B, 1, 1)
                for b in range(B):
                    try:
                        block_11[b, :, :] = torch.linalg.cholesky(RminusPPdivC[b, :, :])
                    except:
                        # keep initialized values (gradient will be close to zero)
                        warnings.warn("backward pass encountered a singular matrix")
                        pass

            block_12 = torch.cholesky_solve(PdivC, block_11)
            #block_22 = torch.diag_embed(1.0 / beta) + torch.bmm(block_12.transpose(1, 2), PdivC)
            block_22 = torch.bmm(block_12.transpose(1, 2), PdivC)

            v1 = torch.cholesky_solve(vHAt1, block_11) - torch.bmm(block_12, vHAt2)
            #v2 = torch.bmm(block_22, vHAt2) - torch.bmm(block_12.transpose(1, 2), vHAt1)
            v2 = vHAt2 / beta.view(B, W, 1) + torch.bmm(block_22, vHAt2) - torch.bmm(block_12.transpose(1, 2), vHAt1)

        else:
            # by full inverse of (A H^{-1] A^T)
            AinvHAt = torch.empty((B, H + W - 1, H + W - 1), device=M.device, dtype=M.dtype)
            AinvHAt[:, 0:H - 1, 0:H - 1] = torch.diag_embed(alpha[:, 1:H])
            AinvHAt[:, H - 1:H + W - 1, H - 1:H + W - 1] = torch.diag_embed(beta)
            AinvHAt[:, 0:H - 1, H - 1:H + W - 1] = P[:, 1:H, 0:W]
            AinvHAt[:, H - 1:H + W - 1, 0:H - 1] = P[:, 1:H, 0:W].transpose(1, 2)

            if ctx.method == 'fullchol':
                v = torch.cholesky_solve(torch.cat((vHAt1, vHAt2), dim=1), torch.linalg.cholesky(AinvHAt))
            else:
                v = torch.bmm(torch.inverse(AinvHAt), torch.cat((vHAt1, vHAt2), dim=1))
                #v = torch.linalg.solve(AinvHAt, torch.cat((vHAt1, vHAt2), dim=1))

            v1 = v[:, 0:H - 1].view(B, H - 1, 1)
            v2 = v[:, H - 1:H + W - 1].view(B, W, 1)

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} A H^{-1} B - v^T H^{-1} B
        dJdM[:, 1:H, 0:W] -= v1 * P[:, 1:H, 0:W]
        dJdM -= v2.view(B, 1, W) * P

        # multiply by derivative of log(M) if in log-space
        if ctx.logspace:
            dJdM /= -1.0 * M

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdr is not None:
            dJdr = ctx.inv_r_sum.view(r.shape[0], 1) / ctx.gamma * \
                   (torch.sum(r[:, 1:H] * v1.view(B, H - 1), dim=1, keepdim=True) - torch.cat((torch.zeros(B, 1, device=r.device), v1.view(B, H - 1)), dim=1))

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdc is not None:
            dJdc = ctx.inv_c_sum.view(c.shape[0], 1) / ctx.gamma * (torch.sum(c * v2.view(B, W), dim=1, keepdim=True) - v2.view(B, W))

        # return gradients (None for gamma, eps, maxiters and logspace)
        return dJdM, dJdr, dJdc, None, None, None, None, None, None


class OptimalTransportLayer(nn.Module):
    """
    Neural network layer to implement optimal transport.

    Parameters:
    -----------
    gamma: float, default: 1.0
        Inverse of the coefficient on the entropy regularisation term.
    eps: float, default: 1.0e-6
        Tolerance used to determine the stop condition.
    maxiters: int, default: 1000
        The maximum number of iterations.
    logspace: bool, default: False
        If `True`, assumes that the input is provided as \log M
        If `False`, assumes that the input is provided as M (standard optimal transport)
    method: str, default: 'block'
        If `approx`, approximate the gradient by assuming exp(-\gamma M) is already nearly doubly stochastic.
        It is faster and could potentially be useful during early stages of training.
        If `block`, exploit the block structure of matrix A H^{-1] A^T.
        If `full`, invert the full A H^{-1} A^T matrix without exploiting the block structure
    """

    def __init__(self, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        super(OptimalTransportLayer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.maxiters = maxiters
        self.logspace = logspace
        self.method = method

    def forward(self, M, r=None, c=None):
        """
        Parameters:
        -----------
        M: torch.Tensor
            Input matrix/matrices of size (H, W) or (B, H, W)
        r: torch.Tensor, optional
            Row sum constraint in the form of a 1xH or BxH matrix. Are assigned uniformly as 1/H by default.
        c: torch.Tensor, optional
            Column sum constraint in the form of a 1xW or BxW matrix. Are assigned uniformly as 1/W by default.

        Returns:
        --------
        torch.Tensor
            Normalised matrix/matrices, with the same shape as the inputs
        """
        M_shape = M.shape
        # Check the number of dimensions
        ndim = len(M_shape)
        if ndim == 2:
            M = M.unsqueeze(dim=0)
        elif ndim != 3:
            raise ValueError(f"The shape of the input tensor {M_shape} does not match that of an matrix")

        # Handle special case of 1x1 matrices
        nr, nc = M_shape[-2:]
        if nr == 1 and nc == 1:
            P = torch.ones_like(M)
        else:
            P = OptimalTransportFcn.apply(M, r, c, self.gamma, self.eps, self.maxiters, self.logspace, self.method)

        return P.view(*M_shape)
