import torch
from LinSATNet import linsat_layer, init_constraints


def init_linsat_constraints(scores, A=None, b=None, C=None, d=None, E=None, f=None):
    constr_dict = init_constraints(scores.shape[-1], A, b, C, d, E, f)
    return constr_dict


def gumbel_linsat_layer(scores, constr_dict,
                        max_iter=100, tau=1., noise_fact=1., sample_num=1000):
    def sample_gumbel(t_like, eps=1e-20):
        """
        randomly sample standard gumbel variables
        """
        u = torch.empty_like(t_like).uniform_()
        return -torch.log(-torch.log(u + eps) + eps)

    s_rep = torch.repeat_interleave(scores.unsqueeze(0), sample_num, dim=0)
    gumbel_noise = sample_gumbel(s_rep) * noise_fact
    s_rep = s_rep + gumbel_noise

    output = linsat_layer(s_rep, constr_dict=constr_dict, tau=tau, max_iter=max_iter, mode='v2', no_warning=False)

    return output
