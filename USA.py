import torch
from torch import nn
import math
import numpy as np



class USA(nn.Module):
    def __init__(self, num_heads, head_dim, usa_params = {'L': 1, 'R': 3, 'int_dim': 128, 'aug_k' : 0}, annealing_paramters = {'T': 10, 't': 0}, use_lsh=False):
        super(USA, self).__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.L = usa_params['L'] #
        self.R = usa_params['R'] #
        self.int_dim = usa_params['int_dim']
        self.aug_k = int(usa_params['aug_k'])
        self.lsh = use_lsh
        if self.lsh:
            self.int_dim = head_dim + 1

        # annealing parameters
        self.T = annealing_paramters['T']
        self.t = annealing_paramters['t']

        # TODO(what is the best way to initialize?)
        # TODO(move to class variables?)
        self.FINAL_DIM = 32
        self.uplift_K = nn.ModuleList([nn.Sequential(nn.Linear(head_dim + self.aug_k, self.int_dim), 
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.FINAL_DIM)
                                    ) for i in range(self.num_heads)])
        self.uplift_Q = nn.ModuleList([nn.Sequential(nn.Linear(head_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.int_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.int_dim, self.FINAL_DIM)
                                      ) for i in range(self.num_heads)])

        self.W = nn.Parameter(torch.randn(self.num_heads, self.int_dim, self.L * self.R), requires_grad=True)
        

    def aug_K(self, K, num):
        if num <= 0:
            return K
        norms = torch.norm(K, dim=-1, keepdim=True)
        aug = []
        for i in range(num):
            aug.append(torch.sqrt(nn.functional.relu(10 * math.pow(10, i) - torch.pow(norms, 2))))
        return torch.cat([K] + aug, dim=-1) / math.sqrt(math.pow(10,num))
        

    def forward(self, K, Q, hard=False):
        b,a,sk,d = K.shape
        _,_,sq,d = Q.shape

        Klifted = torch.zeros((b,a,sk,self.FINAL_DIM), device=K.device)
        Qlifted = torch.zeros((b,a,sq,self.FINAL_DIM), device=Q.device)
        K1 = self.aug_K(K, self.aug_k)
        for i in range(self.num_heads):
            Klifted[:,i,:,:] = self.uplift_K[i](K1[:,i,:,:])
            Qlifted[:,i,:,:] = self.uplift_Q[i](Q[:,i,:,:])

        # get binary hash code
        if hard:
            Klifted = torch.sign(Klifted)
            Qlifted = torch.sign(Qlifted)
        else:
            Klifted = nn.functional.tanh(Klifted)
            Qlifted = nn.functional.tanh(Qlifted)
        
        span =  torch.einsum("basd,batd->bast", Qlifted, Klifted)

        if hard:
            span = (span - self.FINAL_DIM * self.t  > 0).float()
        else:
            span = torch.sigmoid(span - self.FINAL_DIM * self.t)

        return span


    def _forward(self, K, Q, hard=False):

        # shapes
        b,a,sk,d = K.shape
        _,_,sq,d = Q.shape
        idm = self.int_dim
        # uplift K and Q. TODO(try SIMPLE-LSH vs. learned linear projection)
        # common across attention heads
        if not self.lsh:
            Klifted = torch.zeros((b,a,sk,self.int_dim), device=K.device)
            Qlifted = torch.zeros((b,a,sq,self.int_dim), device=Q.device)
            K1 = self.aug_K(K, self.aug_k)
            for i in range(self.num_heads):
                Klifted[:,i,:,:] = self.uplift_K[i](K1[:,i,:,:])
                Qlifted[:,i,:,:] = self.uplift_Q[i](Q[:,i,:,:])
        else:
            M = torch.max(torch.norm(K, dim=-1, keepdim=True), dim=2, keepdim=True).values # b,a,1,1
            Klifted = K / M
            Klifted = torch.cat([Klifted, torch.sqrt(1 - torch.norm(Klifted, dim=-1, keepdim=True))], dim=-1)
            Qlifted =  nn.functional.pad(Q,  (0,1), "constant", 0)

        # soft signed random projections
        if hard or self.lsh:
            Kc = 2* (torch.einsum("basd,ade->base", Klifted, self.W) > 0).float() - 1.  # B,A,S,LR
            Qc = 2* (torch.einsum("basd,ade->base", Qlifted, self.W) > 0).float() - 1.  # B,A,S',LR
        else:
            Kc = nn.functional.tanh(torch.einsum("basd,ade->base", Klifted, self.W) / self.T) # B,A,S,LR
            Qc = nn.functional.tanh(torch.einsum("basd,ade->base", Qlifted, self.W) / self.T) # B,A,S',LR

        # reshape and compute parition match along R
        Kc = Kc.view(b, a, sk, self.L, self.R)
        Qc = Qc.view(b, a, sq, self.L, self.R)
        partition_count = torch.einsum("baslr,batlr->balts", Kc, Qc) / self.R
        
        ## shifted silu since we need all partition to match eventually
        if hard or self.lsh:
            partition_count = ((partition_count - 1.0 + 1e-3) > 0).float()
        else:
            partition_count = nn.functional.sigmoid((partition_count - self.t) / self.T) * partition_count
            
        # Effect of L
        sspan = torch.max(partition_count, dim=2).values # b,a,sk,sq
        return sspan
