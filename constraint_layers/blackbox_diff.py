"""
Modified from: https://github.com/martius-lab/blackbox-backprop/blob/master/blackbox_backprop/ranking.py
"""
import torch


def topk(seq, k):
    """
    :param seq: [BxN] or [N]. The input sequence
    :param k:
    """
    select = torch.zeros_like(seq)
    if len(seq.shape) == 2:
        batch_size = seq.shape[0]
    elif len(seq.shape) == 1:
        batch_size = -1
    else:
        raise ValueError(f'Unknown input shape: {seq.shape}')
    if batch_size >= 0:
        select[
            torch.arange(batch_size).repeat_interleave(k),
            torch.topk(seq, k, dim=-1).indices.view(-1)
        ] = 1
    else:
        select[
            torch.topk(seq, k, dim=-1).indices.view(-1)
        ] = 1
    return select


class BBTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, k, lambda_val):
        select = topk(sequence, k)
        ctx.lambda_val = lambda_val
        ctx.k_val = k
        ctx.save_for_backward(sequence, select)
        return select

    @staticmethod
    def backward(ctx, grad_output):
        sequence, select = ctx.saved_tensors
        assert grad_output.shape == select.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        select_prime = topk(sequence_prime, ctx.k_val)
        gradient = -(select - select_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None, None
