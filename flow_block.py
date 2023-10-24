import torch
import torch.nn as nn
import modules as m

class FlowBlock(nn.Module):
    def __init__(self, z_channels, hidden_layers, hidden_channels, n_steps, permute='inv1x1', condition_channels=None, is_squeeze=True, squzze_type='checkboard',
    is_expand=False, expand_type='checkboard', is_split=False, split_size=None, affine_split=0.5, with_actnorm=True):
        super().__init__()

        self.is_squeeze = is_squeeze
        self.is_expand = is_expand
        self.is_split = is_split

        assert squzze_type == expand_type
        self.squeeze_transition = None
        self.expand_transition = None
        if is_squeeze:
            if squzze_type == 'checkboard':
                self.squeeze = m.CheckboardSqueeze()
                self.squeeze_transition = m.TransitionBlock(z_channels * 4, with_actnorm=with_actnorm)
            elif squzze_type == 'haar':
                self.squeeze = m.HaarWaveletSqueeze(z_channels)
            else:
                raise NotImplemented('Not implemented squeeze type')
        if is_expand:
            if expand_type == 'checkboard':
                self.expand = m.CheckboardExpand()
                self.expand_transition = m.TransitionBlock(z_channels * 4 if self.is_squeeze else z_channels, with_actnorm=with_actnorm)
            elif expand_type == 'haar':
                self.expand = m.HaarWaveletExpand(z_channels if self.is_squeeze else z_channels // 4)
            else:
                raise NotImplemented('Not implemented expand type')

        self.steps = nn.ModuleList()
        if is_squeeze: z_channels = z_channels * 4
        for _ in range(n_steps):
            self.steps.append(m.FlowStep(z_channels, hidden_layers, hidden_channels, permute, condition_channels, affine_split=affine_split, with_actnrom=with_actnorm))
        if is_expand: z_channels = z_channels // 4
        if is_split:
            self.split = m.SplitFlow(z_channels, split_size, n_resblock=8)
    
    def forward(self, z, ldj, tau=0, u=None, reverse=False):
        if not reverse:
            if self.is_squeeze:
                z, ldj = self.squeeze(z, ldj)
                if self.squeeze_transition:
                    z, ldj = self.squeeze_transition(z, ldj)

            for step in self.steps:
                z, ldj = step(z, ldj, u)
            
            if self.is_expand:
                if self.expand_transition:
                    z, ldj = self.expand_transition(z, ldj)
                z, ldj = self.expand(z, ldj)
            
            if self.is_split:
                z, ldj = self.split(z, ldj, tau)

            return z, ldj

        else:
            if self.is_split:
                z, ldj = self.split(z, ldj, tau, reverse=True)
            
            if self.is_expand:
                z, ldj = self.expand(z, ldj, reverse=True)
                if self.expand_transition:
                    z, ldj = self.expand_transition(z, ldj, reverse=True)
            
            for step in self.steps[::-1]:
                z, ldj = step(z, ldj, u, reverse=True)
            
            if self.is_squeeze:
                if self.squeeze_transition:
                    z, ldj = self.squeeze_transition(z, ldj, reverse=True)
                z, ldj = self.squeeze(z, ldj, reverse=True)

            return z, ldj