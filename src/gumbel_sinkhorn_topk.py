from constraint_layers.sinkhorn import Sinkhorn
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def soft_topk(scores, k, max_iter=100, tau=1., return_prob=False):
    return gumbel_sinkhorn_topk(scores, k, max_iter, tau, 0, 1, return_prob)


def gumbel_sinkhorn_topk(scores, k, max_iter=100, tau=1., noise_fact=1., sample_num=1000, return_prob=False):
    anchors = torch.tensor([scores.min(), scores.max()], device=scores.device)
    dist_mat = torch.abs(scores.unsqueeze(-1) - anchors.view(1, 2))

    row_prob = torch.ones(1, scores.shape[0], device=scores.device)
    col_prob = torch.stack(
        (torch.full((1,), scores.shape[0] - k, dtype=torch.float, device=scores.device),
         torch.full((1,), k, dtype=torch.float, device=scores.device),),
        dim=1
    )

    sk = Sinkhorn(max_iter=max_iter, tau=tau, batched_operation=True)

    def sample_gumbel(t_like, eps=1e-20):
        """
        randomly sample standard gumbel variables
        """
        u = torch.empty_like(t_like).uniform_()
        return -torch.log(-torch.log(u + eps) + eps)

    s_rep = torch.repeat_interleave(-dist_mat.unsqueeze(0), sample_num, dim=0)
    gumbel_noise = sample_gumbel(s_rep[:, :, 0]) * noise_fact
    gumbel_noise = torch.stack((gumbel_noise, -gumbel_noise), dim=-1)
    s_rep = s_rep + gumbel_noise
    rows_rep = torch.repeat_interleave(row_prob, sample_num, dim=0)
    cols_rep = torch.repeat_interleave(col_prob, sample_num, dim=0)

    output = sk(s_rep, rows_rep, cols_rep)

    #print(output)
    top_k_indices = torch.topk(output[:, :, 1], k, dim=-1).indices

    if return_prob:
        return top_k_indices, output[:, :, 1]
    else:
        return top_k_indices


if __name__ == '__main__':
    import numpy as np
    a = torch.randn((100,)).cuda()
    fig = plt.figure(figsize=(15, 50), dpi=120)
    enum_list = np.arange(1, -0.1, -0.1)
    for row_id, noise_fact in enumerate(enum_list):
        top_k_indices = gumbel_sinkhorn_topk(a, 10, max_iter=100, tau=0.1, noise_fact=noise_fact)
        plt.subplot(len(enum_list), 2, 1 + row_id*2)
        plt.title(f'noise={noise_fact:.2f}')
        top_k_bincount = torch.bincount(top_k_indices.view(-1), minlength=100)
        sorted_indices = torch.argsort(a.view(-1), descending=True)
        plt.stem(top_k_bincount[sorted_indices].cpu().numpy())
        plt.subplot(len(enum_list), 2, 2 + row_id*2)
        plt.stem(a.view(-1)[sorted_indices].cpu().numpy())
    plt.savefig('gumbel.png', bbox_inches='tight')
