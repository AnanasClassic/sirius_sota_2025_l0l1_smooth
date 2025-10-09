import torch
import math
from torch.optim.optimizer import Optimizer


class AGMsDR_2410_10800(Optimizer):
    def __init__(self, params, L0, L1, stepsize_type='optimal', rel_prec=1e-6, 
                 max_line_search_iters=30, eps=1e-12, dtype=torch.float32):
        """
        Args:
            L0, L1: параметры (L0, L1)-smoothness
            stepsize_type: 'optimal', 'simplified', или 'clipped'
            eps: порог численной стабильности
            dtype: тип данных для вычислений
        """
        defaults = dict(L0=L0, L1=L1, stepsize_type=stepsize_type, rel_prec=rel_prec, 
                       max_line_search_iters=max_line_search_iters, eps=eps, dtype=dtype)
        super(AGMsDR_2410_10800, self).__init__(params, defaults)
    
    def step(self, closure):
        """closure должен принимать backward=True/False"""
        if closure is None:
            raise RuntimeError("AGMsDR_2410_10800 требует closure")
        
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
            
            y_k = self._line_search(params, v_k, x_k, closure, rel_prec)
            
            with torch.no_grad():
                for p, y in zip(params, y_k):
                    p.data.copy_(y)
            
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            loss_y = closure(backward=True)
            grad_y = [p.grad.clone() if p.grad is not None else torch.zeros_like(p.data) for p in params]
            grad_norm_sq = sum(torch.sum(g ** 2) for g in grad_y)
            grad_norm = torch.sqrt(grad_norm_sq).item()
            
            if grad_norm < eps:
                loss = loss_y
                state['step'] += 1
                continue
            
            with torch.no_grad():
                eta = self._compute_stepsize(grad_norm, L0, L1, stepsize_type, eps)
                state['last_stepsize'] = eta
                x_k_new = [y - eta * g for y, g in zip(y_k, grad_y)]
                
                for p, x_new in zip(params, x_k_new):
                    p.data.copy_(x_new)
            
            loss_x = closure(backward=False)
            
            with torch.no_grad():
                delta_f = (loss_y - loss_x).item()
                
                if delta_f > eps and grad_norm > eps:
                    M_k = min(grad_norm ** 2 / (2 * delta_f), 1e10)
                    
                    a_k = self._find_a_k(M_k, A_k, max_a=1e6)
                    
                    if a_k > eps:
                        A_k = min(A_k + a_k, 1e8)
                        
                        v_k = [v - a_k * g for v, g in zip(v_k, grad_y)]
                
                state['v'] = v_k
                state['A'] = A_k
                state['step'] += 1
            
            loss = loss_x
        
        return loss
    
    def _compute_stepsize(self, grad_norm, L0, L1, stepsize_type, eps):
        
        if stepsize_type == 'optimal':
            if L1 > 0 and grad_norm > eps:
                ratio = L1 * grad_norm / (L0 + L1 * grad_norm)
                if ratio > 100:
                    eta = 1.0 / (L1 * grad_norm)
                else:
                    eta = math.log(1 + ratio) / (L1 * grad_norm)
            else:
                eta = 1.0 / max(L0, eps)
        
        elif stepsize_type == 'simplified':
            eta = 1.0 / max(L0 + 1.5 * L1 * grad_norm, eps)
        
        elif stepsize_type == 'clipped':
            eta = min(1.0 / max(2 * L0, eps), 
                     1.0 / max(3 * L1 * grad_norm, eps))
        
        else:
            raise ValueError(f"Unknown stepsize_type: {stepsize_type}")
        
        eta = max(min(eta, 1e6), eps)
        return float(eta)
    
    def _line_search(self, params, v_k, x_k, closure, rel_prec):
        direction = [x - v for x, v in zip(x_k, v_k)]
        beta_left = 0.0
        beta_right = 1.0
        
        orig_data = [p.data.clone() for p in params]
        
        max_iters = 50
        iter_count = 0
        
        while (beta_right - beta_left > rel_prec) and (iter_count < max_iters):
            beta1 = beta_left + (beta_right - beta_left) / 3
            beta2 = beta_right - (beta_right - beta_left) / 3
            
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta1 * d)
            
            loss1 = closure(backward=False).item()
            
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta2 * d)
            
            loss2 = closure(backward=False).item()
            
            if loss1 < loss2:
                beta_right = beta2
            else:
                beta_left = beta1
            
            iter_count += 1
        
        with torch.no_grad():
            for p, orig in zip(params, orig_data):
                p.data.copy_(orig)
        
        beta_opt = (beta_left + beta_right) / 2
        return [v + beta_opt * d for v, d in zip(v_k, direction)]
    
    def _find_a_k(self, M_k, A_k, max_a=1e6):
        if M_k <= 0:
            return 0.0
        
        product = 4 * M_k * A_k
        
        if product > 1e10:
            a_k = math.sqrt(A_k / M_k)
        else:
            discriminant = 1 + product
            if discriminant < 0:
                return 0.0
            
            sqrt_disc = math.sqrt(discriminant)
            
            if A_k > 0:
                a_k = 2 * A_k / (1 + sqrt_disc)
            else:
                a_k = (sqrt_disc - 1) / (2 * M_k)
        
        a_k = min(max(a_k, 0.0), max_a)
        return float(a_k)
    
import torch
from torch.optim.optimizer import Optimizer
import math


class AGMsDR_1809_05895(Optimizer):
    """
    Реализация Algorithm 1 (AGMsDR) из arXiv:1809.05895 (вариант с известным L, Option a).
    Требует closure, который при вызове closure(backward=True) вычисляет loss и делает backward(),
    а при closure(backward=False) возвращает значение функции (без backward).
    """
    def __init__(self, params, L, rel_prec=1e-6, max_line_search_iters=50, eps=1e-12, dtype=torch.float32):
        if L is None:
            raise ValueError("L (Lipschitz constant) must be provided for this implementation (Option a).")
        defaults = dict(L=L, rel_prec=rel_prec,
                        max_line_search_iters=max_line_search_iters, eps=eps, dtype=dtype)
        super(AGMsDR_1809_05895, self).__init__(params, defaults)

    def step(self, closure):
        if closure is None:
            raise RuntimeError("AGMsDR_1809_05895 requires closure")

        loss = None

        for group in self.param_groups:
            L = float(group['L'])
            rel_prec = group['rel_prec']
            max_line_search_iters = group['max_line_search_iters']
            eps = group['eps']

            state = self.state.setdefault('global', {})

            params = group['params']

            # Инициализация состояния (один раз)
            if 'initialized' not in state:
                state['step'] = 0
                state['A'] = 0.0                     # A_k
                # x0 — фиксированная начальная точка (список тензоров)
                state['x0'] = [p.data.clone().detach() for p in params]
                # v = v_k = x0 - sum a_i grad(y_i). Инициално v0 = x0
                state['v'] = [p.data.clone().detach() for p in params]
                # psi_linear = sum_i a_i * grad_yi  (вектора)
                state['psi_linear'] = [torch.zeros_like(p.data) for p in params]
                # psi_const = сумма постоянных членов (скаляр)
                state['psi_const'] = 0.0
                state['initialized'] = True

            # Переменные шага
            A_k = state['A']
            v_k = [t.clone().detach() for t in state['v']]            # список тензоров
            psi_linear = [t.clone().detach() for t in state['psi_linear']]
            psi_const = float(state['psi_const'])
            x0 = [t.clone().detach() for t in state['x0']]

            # текущие параметры x_k (считываем из params)
            x_k = [p.data.clone().detach() for p in params]

            # ========== line-search для beta: найти y_k = v_k + beta*(x_k - v_k), beta in [0,1] ==========
            y_k = self._line_search_beta(params, v_k, x_k, closure, rel_prec, max_line_search_iters)

            # Установить параметры в y_k (temporarily) для вычисления градиента
            with torch.no_grad():
                for p, y in zip(params, y_k):
                    p.data.copy_(y)

            # Обнулить градиенты перед вызовом closure
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            # Вычислить loss в y_k и сделать backward чтобы получить grad_y
            loss_y = closure(backward=True)
            # Скопировать градиенты (все как отдельные тензоры)
            grad_y = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p.data)
                      for p in params]

            # ========== градиентный шаг (Option a, L известен) ==========
            if L <= eps:
                raise ValueError("L must be positive and > eps")

            with torch.no_grad():
                x_k_plus = [y - (1.0 / L) * g for y, g in zip(y_k, grad_y)]
                for p, x_new in zip(params, x_k_plus):
                    p.data.copy_(x_new)

            # вычислить функцию в x_{k+1} (без backward)
            loss_x = closure(backward=False)

            # ========== найти a_{k+1} по квадратному уравнению: a^2/(A_k + a) = 1/L ==========
            a_k = self._solve_a(A_k, L, eps)
            A_k_new = A_k + a_k

            # ========== обновить psi (psi_linear и psi_const) ==========
            # psi_const stores sum_i a_i * ( f(y_i) - <grad_yi, y_i> )
            # psi_linear stores sum_i a_i * grad_yi
            with torch.no_grad():
                # фрагмент <g, y> суммируем в скаляре
                sum_g_dot_y = 0.0
                for g, y in zip(grad_y, y_k):
                    sum_g_dot_y += float(torch.sum(g * y).item())

                psi_const += a_k * (float(loss_y) - sum_g_dot_y)

                psi_linear = [pl + a_k * g for pl, g in zip(psi_linear, grad_y)]

            # ========== обновить v_{k+1} = argmin_x psi_{k+1}(x) = x0 - psi_linear ==========
            with torch.no_grad():
                v_k_new = [x0_i - pl for x0_i, pl in zip(x0, psi_linear)]

            # Сохранить новое состояние
            state['v'] = [t.clone().detach() for t in v_k_new]
            state['A'] = A_k_new
            state['psi_linear'] = [t.clone().detach() for t in psi_linear]
            state['psi_const'] = float(psi_const)
            state['step'] += 1
            state['last_a'] = float(a_k)

            # Сообщаем loss на выход (значение в x_{k+1})
            loss = loss_x

        return loss

    def _line_search_beta(self, params, v_k, x_k, closure, rel_prec, max_iters):
        """
        Нахождение y_k = v_k + beta*(x_k - v_k), beta in [0,1], минимизирующее f.
        Реализовано через ternary search (подходит, т.к. поиск по отрезку).
        Возвращает список тензоров y (не меняет params в конце — восстанавливает).
        """
        # направление d = x_k - v_k
        direction = [x - v for x, v in zip(x_k, v_k)]
        beta_left = 0.0
        beta_right = 1.0

        # Сохраняем оригинальные параметры, чтобы восстановить после line-search
        orig_data = [p.data.clone().detach() for p in params]

        iter_count = 0
        while (beta_right - beta_left > rel_prec) and (iter_count < max_iters):
            beta1 = beta_left + (beta_right - beta_left) / 3.0
            beta2 = beta_right - (beta_right - beta_left) / 3.0

            # f(v + beta1 * d)
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta1 * d)
            loss1 = float(closure(backward=False))

            # f(v + beta2 * d)
            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta2 * d)
            loss2 = float(closure(backward=False))

            if loss1 < loss2:
                beta_right = beta2
            else:
                beta_left = beta1

            iter_count += 1

        beta_opt = 0.5 * (beta_left + beta_right)

        # восстановить оригинальные параметры
        with torch.no_grad():
            for p, orig in zip(params, orig_data):
                p.data.copy_(orig)

        # вернуть y = v + beta_opt * d (список тензоров)
        y = [v + beta_opt * d for v, d in zip(v_k, direction)]
        return y

    def _solve_a(self, A_k, L, eps):
        """
        Решение для a: a^2/(A_k + a) = 1/L, взять положительное решение (больший корень).
        Эквивалентно: L a^2 - a - A_k = 0.
        Решение: a = (1 + sqrt(1 + 4 L A_k)) / (2 L)
        """
        if L <= eps:
            return 0.0
        discriminant = 1.0 + 4.0 * L * A_k
        if discriminant < 0.0:
            return 0.0
        a = (1.0 + math.sqrt(discriminant)) / (2.0 * L)
        return max(a, 0.0)


class AGMsDR_1809_05895_Coordinate(Optimizer):
    """
    Координатная версия AGMsDR Algorithm 1 из arXiv:1809.05895 (Option a).
    На каждой итерации случайно выбирается одна координата, а шаг выполняется только по ней.
    """

    def __init__(self, params, L, rel_prec=1e-6, max_line_search_iters=50, eps=1e-12, seed=None):
        if L is None:
            raise ValueError("L (Lipschitz constant) must be provided for this coordinate version.")
        defaults = dict(L=L, rel_prec=rel_prec,
                        max_line_search_iters=max_line_search_iters,
                        eps=eps, seed=seed)
        super(AGMsDR_1809_05895_Coordinate, self).__init__(params, defaults)

        if seed is not None:
            torch.manual_seed(seed)

    def step(self, closure):
        if closure is None:
            raise RuntimeError("AGMsDR_1809_05895_Coordinate requires closure")

        loss = None

        for group in self.param_groups:
            L = float(group['L'])
            rel_prec = group['rel_prec']
            max_line_search_iters = group['max_line_search_iters']
            eps = group['eps']

            state = self.state.setdefault('global', {})

            params = group['params']

            # --- инициализация ---
            if 'initialized' not in state:
                state['step'] = 0
                state['A'] = 0.0
                state['x0'] = [p.data.clone().detach() for p in params]
                state['v'] = [p.data.clone().detach() for p in params]
                state['psi_linear'] = [torch.zeros_like(p.data) for p in params]
                state['psi_const'] = 0.0
                state['total_coords'] = sum(p.numel() for p in params)
                state['initialized'] = True

            A_k = state['A']
            v_k = [t.clone().detach() for t in state['v']]
            psi_linear = [t.clone().detach() for t in state['psi_linear']]
            psi_const = float(state['psi_const'])
            x0 = [t.clone().detach() for t in state['x0']]
            total_coords = int(state['total_coords'])

            x_k = [p.data.clone().detach() for p in params]

            # --- поиск beta для y_k ---
            y_k = self._line_search_beta(params, v_k, x_k, closure, rel_prec, max_line_search_iters)

            with torch.no_grad():
                for p, y in zip(params, y_k):
                    p.data.copy_(y)

            # --- вычислить градиент в y_k ---
            for p in params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            loss_y = closure(backward=True)
            full_grad_y = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p.data)
                           for p in params]

            # --- выбрать случайную координату ---
            coord_idx = torch.randint(0, total_coords, (1,)).item()

            grad_y = []
            current_idx = 0
            for full_g in full_grad_y:
                param_size = full_g.numel()
                coord_grad = torch.zeros_like(full_g)

                if current_idx <= coord_idx < current_idx + param_size:
                    local_idx = coord_idx - current_idx
                    flat_grad = full_g.view(-1)
                    # Несмещённая оценка: умножаем на n (число координат)
                    coord_grad.view(-1)[local_idx] = flat_grad[local_idx] * total_coords

                grad_y.append(coord_grad)
                current_idx += param_size

            # --- градиентный шаг по выбранной координате ---
            L_coord = L * total_coords  # разумное приближение для coordinate smoothness

            if L_coord <= eps:
                raise ValueError("L_coord must be positive.")

            with torch.no_grad():
                x_k_new = [y - (1.0 / L_coord) * g for y, g in zip(y_k, grad_y)]
                for p, x_new in zip(params, x_k_new):
                    p.data.copy_(x_new)

            loss_x = closure(backward=False)

            # --- решить для a_k ---
            a_k = self._solve_a(A_k, L_coord, eps)
            A_k_new = A_k + a_k

            # --- обновить psi ---
            with torch.no_grad():
                sum_g_dot_y = 0.0
                for g, y in zip(grad_y, y_k):
                    sum_g_dot_y += float(torch.sum(g * y).item())

                psi_const += a_k * (float(loss_y) - sum_g_dot_y)
                psi_linear = [pl + a_k * g for pl, g in zip(psi_linear, grad_y)]

            # --- обновить v_k ---
            with torch.no_grad():
                v_k_new = [x0_i - pl for x0_i, pl in zip(x0, psi_linear)]

            # --- сохранить состояние ---
            state['v'] = [t.clone().detach() for t in v_k_new]
            state['A'] = A_k_new
            state['psi_linear'] = [t.clone().detach() for t in psi_linear]
            state['psi_const'] = float(psi_const)
            state['step'] += 1
            state['last_a'] = float(a_k)

            loss = loss_x

        return loss

    # --- line search по beta (та же логика, что и в полной версии) ---
    def _line_search_beta(self, params, v_k, x_k, closure, rel_prec, max_iters):
        direction = [x - v for x, v in zip(x_k, v_k)]
        beta_left = 0.0
        beta_right = 1.0
        orig_data = [p.data.clone().detach() for p in params]
        iter_count = 0

        while (beta_right - beta_left > rel_prec) and (iter_count < max_iters):
            beta1 = beta_left + (beta_right - beta_left) / 3.0
            beta2 = beta_right - (beta_right - beta_left) / 3.0

            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta1 * d)
            loss1 = float(closure(backward=False))

            with torch.no_grad():
                for p, v, d in zip(params, v_k, direction):
                    p.data.copy_(v + beta2 * d)
            loss2 = float(closure(backward=False))

            if loss1 < loss2:
                beta_right = beta2
            else:
                beta_left = beta1

            iter_count += 1

        beta_opt = 0.5 * (beta_left + beta_right)

        with torch.no_grad():
            for p, orig in zip(params, orig_data):
                p.data.copy_(orig)

        return [v + beta_opt * d for v, d in zip(v_k, direction)]

    # --- решение уравнения для a_k ---
    def _solve_a(self, A_k, L, eps):
        if L <= eps:
            return 0.0
        disc = 1.0 + 4.0 * L * A_k
        if disc < 0.0:
            return 0.0
        a = (1.0 + math.sqrt(disc)) / (2.0 * L)
        return max(a, 0.0)
