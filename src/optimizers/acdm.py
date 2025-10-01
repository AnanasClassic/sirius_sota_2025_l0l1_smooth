import torch
from torch.optim.optimizer import Optimizer


class ACDM(Optimizer):
    def __init__(self, params, L, alpha=1.0, sigma=0.0):
        if not isinstance(L, (list, tuple, torch.Tensor)):
            raise ValueError("L must be a list, tuple, or tensor")
        
        L_tensor = torch.tensor(L, dtype=torch.float32) if not isinstance(L, torch.Tensor) else L
        
        if torch.any(L_tensor <= 0):
            raise ValueError("All L values must be positive")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        
        defaults = dict(L=L_tensor, alpha=alpha, sigma=sigma)
        super(ACDM, self).__init__(params, defaults)
        
        self.alpha = alpha
        self.beta = alpha / 2.0
        self.sigma = sigma
        self.n = len(L_tensor)
        self.L = L_tensor
        
        S_beta = torch.sum(self.L ** self.beta)
        if S_beta <= 0:
            raise ValueError("Sum of L^beta nonpositive")
        self.S_beta = S_beta
        self.pi = (self.L ** self.beta) / S_beta
    
    def _sample_coordinate(self):
        return torch.multinomial(self.pi, 1).item()
    
    def _solve_for_a(self, A_t, B_t):
        s = self.sigma ** (1.0 - self.alpha) if self.sigma > 0 else 0.0
        S2 = self.S_beta ** 2
        C2 = S2 - s
        C1 = -(A_t * s + B_t)
        C0 = -A_t * B_t
        
        if abs(C2) < 1e-60:
            if abs(C1) < 1e-60:
                return 1.0 / self.S_beta
            a = -C0 / C1
            return max(a, 1e-12)
        
        disc = C1 * C1 - 4.0 * C2 * C0
        if disc < 0:
            disc = 0.0
        sqrt_disc = torch.sqrt(torch.tensor(disc))
        
        r1 = (-C1 + sqrt_disc) / (2.0 * C2)
        r2 = (-C1 - sqrt_disc) / (2.0 * C2)
        
        a = None
        for r in (r1, r2):
            r_val = r.item() if isinstance(r, torch.Tensor) else r
            if r_val > 0 and torch.isfinite(torch.tensor(r_val)):
                if a is None or r_val < a:
                    a = r_val
        if a is None:
            a = 1.0 / self.S_beta.item()
        return a
    
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("ACDM requires closure")
        
        for group in self.param_groups:
            L = group['L']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['v'] = p.data.clone()
                    state['A'] = 0.0
                    state['B'] = 1.0
                
                x = p.data
                v = state['v']
                A_t = state['A']
                B_t = state['B']
                
                i_t = self._sample_coordinate()
                
                a_tp1 = self._solve_for_a(A_t, B_t)
                A_tp1 = A_t + a_tp1
                s = self.sigma ** (1.0 - self.alpha) if self.sigma > 0 else 0.0
                B_tp1 = B_t + s * a_tp1
                
                alpha_t = a_tp1 / A_tp1
                beta_t = (s * a_tp1) / B_tp1 if B_tp1 != 0 else 0.0
                
                denom = 1.0 - alpha_t * beta_t
                if abs(denom) < 1e-16:
                    denom = 1e-16
                
                y = ((1.0 - alpha_t) * x + alpha_t * (1.0 - beta_t) * v) / denom
                
                p.data = y
                loss = closure()
                loss.backward()
                g = p.grad.clone()
                p.grad.zero_()
                
                Li = L[i_t]
                x_next = y.clone()
                x_next.view(-1)[i_t] = y.view(-1)[i_t] - g.view(-1)[i_t] / Li
                
                denom_factor = (L[i_t] ** (1.0 - self.alpha)) * B_tp1 * self.pi[i_t]
                if denom_factor == 0:
                    add = torch.zeros_like(v)
                else:
                    coeff = a_tp1 / denom_factor
                    add = torch.zeros_like(v)
                    add.view(-1)[i_t] = coeff * (-g.view(-1)[i_t])
                
                v_next = (1.0 - beta_t) * v + beta_t * y + add
                
                p.data = x_next
                state['v'] = v_next
                state['A'] = A_tp1
                state['B'] = B_tp1
        
        return loss