import torch
import math
from torch.optim.optimizer import Optimizer


class AGMsDR_2410_10800(Optimizer):
    def __init__(self, params, L0, L1, stepsize_type='optimal', rel_prec=1e-6, 
                 max_line_search_iters=30, eps=1e-12):
        """
        Args:
            L0, L1: параметры (L0, L1)-smoothness
            stepsize_type: 'optimal', 'simplified', или 'clipped'
            eps: порог численной стабильности
        """
        defaults = dict(L0=L0, L1=L1, stepsize_type=stepsize_type, rel_prec=rel_prec, 
                       max_line_search_iters=max_line_search_iters, eps=eps)
        super(AGMsDR, self).__init__(params, defaults)
    
    def step(self, closure):
        """closure должен принимать backward=True/False"""
        if closure is None:
            raise RuntimeError("AGMsDR требует closure")
        
        loss = None
        
        for group in self.param_groups:
            L0 = group['L0']
            L1 = group['L1']
            stepsize_type = group['stepsize_type']
            rel_prec = group['rel_prec']
            eps = group['eps']
            
            state = self.state.setdefault('global', {})
            
            if 'step' not in state:
                state['step'] = 0
                state['v'] = [p.data.clone() for p in group['params']]
                state['A'] = 0.0
            
            params = group['params']
            v_k = state['v']
            A_k = state['A']
            x_k = [p.data.clone() for p in params]
            
            # Шаг 4: Line search для y_k
            y_k = self._line_search(params, v_k, x_k, closure, rel_prec)
            
            # Устанавливаем y_k
            with torch.no_grad():
                for p, y in zip(params, y_k):
                    p.data.copy_(y)
            
            # Вычисляем f(y_k) и ∇f(y_k)
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            loss_y = closure(backward=True)
            grad_y = [p.grad.clone() for p in params]
            grad_norm_sq = sum(torch.sum(g ** 2) for g in grad_y)
            grad_norm = torch.sqrt(grad_norm_sq).item()
            
            # КРИТИЧНО: Проверка на слишком малый градиент
            if grad_norm < eps:
                # Градиент почти нулевой - сходимость достигнута
                loss = loss_y
                state['step'] += 1
                continue
            
            # Шаг 5: Градиентный шаг с правильным степсайзом
            with torch.no_grad():
                eta = self._compute_stepsize(grad_norm, L0, L1, stepsize_type, eps)
                # Сохраняем step_size для отображения
                state['last_stepsize'] = eta
                x_k_new = [y - eta * g for y, g in zip(y_k, grad_y)]
                
                for p, x_new in zip(params, x_k_new):
                    p.data.copy_(x_new)
            
            # Вычисляем f(x_{k+1})
            loss_x = closure(backward=False)
            
            with torch.no_grad():
                # Вычисляем M_k
                delta_f = (loss_y - loss_x).item()
                
                # ИСПРАВЛЕНИЕ: Обновляем v только если есть значимое улучшение
                if delta_f > eps and grad_norm > eps:
                    # Ограничиваем M_k для численной стабильности
                    M_k = min(grad_norm ** 2 / (2 * delta_f), 1e10)
                    
                    # Находим a_{k+1} с защитой от переполнения
                    a_k = self._find_a_k(M_k, A_k, max_a=1e6)
                    
                    if a_k > eps:
                        # ИСПРАВЛЕНИЕ: Ограничиваем рост A_k
                        A_k = min(A_k + a_k, 1e8)
                        
                        # Обновляем v
                        v_k = [v - a_k * g for v, g in zip(v_k, grad_y)]
                
                state['v'] = v_k
                state['A'] = A_k
                state['step'] += 1
            
            loss = loss_x
        
        return loss
    
    def _compute_stepsize(self, grad_norm, L0, L1, stepsize_type, eps):
        """Вычисляет степсайз согласно формулам (3.2), (3.5) или (3.6)"""
        
        if stepsize_type == 'optimal':
            # Формула (3.2) с защитой от переполнения
            if L1 > 0 and grad_norm > eps:
                ratio = L1 * grad_norm / (L0 + L1 * grad_norm)
                # Защита от log(1 + очень большое число)
                if ratio > 100:
                    eta = 1.0 / (L1 * grad_norm)
                else:
                    eta = math.log(1 + ratio) / (L1 * grad_norm)
            else:
                eta = 1.0 / max(L0, eps)
        
        elif stepsize_type == 'simplified':
            # Формула (3.5)
            eta = 1.0 / max(L0 + 1.5 * L1 * grad_norm, eps)
        
        elif stepsize_type == 'clipped':
            # Формула (3.6)
            eta = min(1.0 / max(2 * L0, eps), 
                     1.0 / max(3 * L1 * grad_norm, eps))
        
        else:
            raise ValueError(f"Unknown stepsize_type: {stepsize_type}")
        
        # Ограничиваем степсайз разумными пределами
        eta = max(min(eta, 1e6), eps)
        return float(eta)
    
    def _line_search(self, params, v_k, x_k, closure, rel_prec):
        """Тернарный поиск на отрезке [v_k, x_k]"""
        direction = [x - v for x, v in zip(x_k, v_k)]
        beta_left = 0.0
        beta_right = 1.0
        
        orig_data = [p.data.clone() for p in params]
        
        max_iters = 50  # Ограничение на количество итераций
        iter_count = 0
        
        while (beta_right - beta_left > rel_prec) and (iter_count < max_iters):
            beta1 = beta_left + (beta_right - beta_left) / 3
            beta2 = beta_right - (beta_right - beta_left) / 3
            
            # Оцениваем f(β1)
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta1 * d)
            
            loss1 = closure(backward=False).item()
            
            # Оцениваем f(β2)
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta2 * d)
            
            loss2 = closure(backward=False).item()
            
            if loss1 < loss2:
                beta_right = beta2
            else:
                beta_left = beta1
            
            iter_count += 1
        
        # Восстанавливаем параметры
        with torch.no_grad():
            for p, orig in zip(params, orig_data):
                p.data.copy_(orig)
        
        beta_opt = (beta_left + beta_right) / 2
        return [v + beta_opt * d for v, d in zip(v_k, direction)]
    
    def _find_a_k(self, M_k, A_k, max_a=1e6):
        """Решает M_k * a² = A_k + a с защитой от переполнения"""
        if M_k <= 0:
            return 0.0
        
        # ИСПРАВЛЕНИЕ: Используем более стабильную формулу
        # a = (-1 + sqrt(1 + 4*M_k*A_k)) / (2*M_k)
        # Переписываем для избежания переполнения:
        # a = 2*A_k / (1 + sqrt(1 + 4*M_k*A_k))
        
        product = 4 * M_k * A_k
        
        # Если произведение слишком велико, используем приближение
        if product > 1e10:
            # sqrt(1 + x) ≈ sqrt(x) для больших x
            # a ≈ 2*A_k / (2*sqrt(M_k*A_k)) = sqrt(A_k/M_k)
            a_k = math.sqrt(A_k / M_k)
        else:
            discriminant = 1 + product
            if discriminant < 0:
                return 0.0
            
            sqrt_disc = math.sqrt(discriminant)
            
            # Используем более стабильную формулу
            if A_k > 0:
                a_k = 2 * A_k / (1 + sqrt_disc)
            else:
                a_k = (sqrt_disc - 1) / (2 * M_k)
        
        # Ограничиваем a_k
        a_k = min(max(a_k, 0.0), max_a)
        return float(a_k)