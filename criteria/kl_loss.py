import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, x_q, m_q, logs_q, z_p_, z_p, m_p, logs_p, x_p, logdet):  # latent, x_q_mean, x_q_var, z_p_, z_p, z_p_mean, z_p_var, x_p_codes, logdet
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """

        kl0 = F.l1_loss(x_p, z_p)
        kl1 = F.l1_loss(z_p, z_p_.detach())
        kl2 = 0.5 * (torch.log(logs_p) - torch.log(logs_q).detach() - 1)
        kl3 = (m_q**2 + logs_q - 2 * m_q * m_p + m_p**2) / (2 * logs_p)
        kl4 = torch.abs(logdet)
        kl = kl0 + kl1 + torch.abs(torch.mean(kl2) + torch.mean(kl3) + torch.mean(kl4))
        print('kl0', kl0)
        print('kl1', kl1)
        print('kl2', kl2)
        print('kl3', kl3)
        print('kl4', kl4)

        return kl