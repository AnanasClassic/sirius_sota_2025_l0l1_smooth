import numpy as np
import torch
from tqdm import tqdm


class CDM:
    def __init__(self, x, f, grad_f, L0, L1, dtype=torch.float32):
        self.dtype = dtype
        self.x_k = x.clone()
        self.L0 = L0
        self.dim = len(self.x_k)
        self.L1 = L1
        self.f = f
        self.grad_f = grad_f
        self.history = []

    def cycle(self, num_iter, seed=None, tol=1e-18, verbose=False):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        for k in tqdm(range(num_iter), disable=not verbose):
            i = np.random.randint(0, self.dim)

            g = self.grad_f(self.x_k)
            nu = 1 / (self.L0 + self.L1 * torch.abs(g[i]))

            self.x_k[i] = self.x_k[i] - nu * g[i]

            self.history.append(self.f(self.x_k))
        return self.history