import torch
from torch.optim.optimizer import Optimizer


class AGMsDR(Optimizer):
    def __init__(self, params, lr=1.0, rel_prec=1e-6):
        defaults = dict(lr=lr, rel_prec=rel_prec)
        super(AGMsDR, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure):
        """closure должен возвращать loss"""
        if closure is None:
            raise RuntimeError("AGMsDR требует closure")
        
        for group in self.param_groups:
            lr = group['lr']
            rel_prec = group['rel_prec']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Инициализация состояния
                if len(state) == 0:
                    state['v'] = p.data.clone()
                    state['A'] = 0.0
                
                x_k = p.data
                v_k = state['v']
                A_k = state['A']
                
                # Тернарный поиск y_k между x_k и v_k
                y_k = self._ternary_search(x_k, v_k, closure, rel_prec)
                
                # Вычисляем градиент в y_k
                p.data = y_k
                loss_y = closure()
                loss_y.backward()
                grad_y = p.grad.clone()
                p.grad.zero_()
                
                # Обновление x_k по правилу (в данном случае градиентный шаг)
                x_k_new = y_k - lr * grad_y
                p.data = x_k_new
                loss_x = closure()
                
                # Вычисляем M_k
                grad_norm_sq = torch.sum(grad_y ** 2)
                delta_f = loss_y - loss_x
                
                if delta_f > 0:
                    M_k = grad_norm_sq / (2 * delta_f)
                    
                    # Находим a_k из уравнения M_k * a_k^2 = A_k + a_k
                    a_k = self._find_a_k(M_k, A_k)
                    
                    if a_k is not None and a_k > 0:
                        A_k += a_k
                        v_k = v_k - a_k * grad_y
                
                state['v'] = v_k
                state['A'] = A_k
        
        return loss_x
    
    def _ternary_search(self, a, b, closure, rel_prec):
        """Тернарный поиск минимума между a и b"""
        tol = rel_prec * torch.norm(b - a)
        if tol == 0:
            return (a + b) / 2
        
        a_curr = a.clone()
        b_curr = b.clone()
        
        while torch.norm(b_curr - a_curr) > tol:
            m1 = a_curr + (b_curr - a_curr) / 3
            m2 = b_curr - (b_curr - a_curr) / 3
            
            # Сохраняем текущие параметры
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    orig_data = p.data.clone()
                    
                    p.data = m1
                    loss1 = closure()
                    
                    p.data = m2
                    loss2 = closure()
                    
                    p.data = orig_data
                    
                    if loss1 < loss2:
                        b_curr = m2
                    else:
                        a_curr = m1
                    break
                break
        
        return (a_curr + b_curr) / 2
    
    def _find_a_k(self, M_k, A_k):
        """Решает уравнение M_k * a_k^2 = A_k + a_k"""
        D = 1 + 4 * M_k * A_k
        if D < 0:
            return None
        
        sqrt_D = torch.sqrt(D)
        a1 = (1 - sqrt_D) / (2 * M_k)
        a2 = (1 + sqrt_D) / (2 * M_k)
        
        return torch.max(torch.tensor([a1, a2, 0.0]))

