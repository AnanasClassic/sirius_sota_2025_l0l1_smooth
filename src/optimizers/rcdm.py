import torch
from torch.optim.optimizer import Optimizer


class RCDM(Optimizer):
    """
    Random Coordinate Descent Method (RCDM) для безусловной минимизации.
    
    Алгоритм выбирает случайную координату на каждой итерации и обновляет только её,
    используя вероятности, пропорциональные константам Липшица в степени alpha.
    """
    
    def __init__(self, params, L=1.0, alpha=1.0, eps=1e-12, seed=None, dtype=torch.float32):
        """
        Args:
            params: model parameters
            L: Липшицева константа (скаляр или тензор размерности параметров)
            alpha: параметр α ∈ R для вычисления вероятностей (формула 2.5)
            eps: порог численной стабильности
            seed: random seed для воспроизводимости
            dtype: тип данных для вычислений
        """
        defaults = dict(L=L, alpha=alpha, eps=eps, seed=seed, dtype=dtype)
        super(RCDM, self).__init__(params, defaults)
        
        if seed is not None:
            torch.manual_seed(seed)
    
    def step(self, closure):
        """
        Выполняет один шаг оптимизации.
        
        Args:
            closure: функция, вычисляющая loss и вызывающая backward
        
        Returns:
            loss value
        """
        if closure is None:
            raise RuntimeError("RCDM требует closure")
        
        loss = None
        
        for group in self.param_groups:
            L = group['L']
            alpha = group['alpha']
            eps = group['eps']
            
            state = self.state.setdefault('global', {})
            
            if 'step' not in state:
                state['step'] = 0
                # Вычисляем вероятности выбора координат и общее количество координат
                state['total_coords'] = sum(p.numel() for p in group['params'])
                state['probabilities'] = self._compute_probabilities(group['params'], L, alpha, eps)
            
            params = group['params']
            
            # Очищаем градиенты
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            
            # Вычисляем функцию и градиент
            loss = closure(backward=True)
            
            # Получаем полные градиенты
            full_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p.data) 
                         for p in params]
            
            # Выбираем случайную координату согласно вероятностям
            coord_idx = self._sample_coordinate(state['probabilities'])
            
            # Находим параметр и локальный индекс выбранной координаты
            param_idx, local_idx = self._find_param_and_local_idx(params, coord_idx)
            
            # Получаем константу Липшица для этой координаты
            L_i = self._get_coordinate_lipschitz(params, L, param_idx, local_idx, eps)
            
            # Обновляем только выбранную координату
            # x_{k+1}[i_k] = x_k[i_k] - (1/L_{i_k}) * grad[i_k]
            with torch.no_grad():
                p = params[param_idx]
                flat_p = p.data.view(-1)
                flat_grad = full_grads[param_idx].view(-1)
                
                # Обновление выбранной координаты
                flat_p[local_idx] -= (1.0 / L_i) * flat_grad[local_idx]
            
            state['step'] += 1
        
        return loss
    
    def _compute_probabilities(self, params, L, alpha, eps):
        """
        Вычисляет вероятности выбора координат по формуле (2.5):
        p_i^(α) = L_i^α / sum_j(L_j^α)
        """
        total_coords = sum(p.numel() for p in params)
        
        if isinstance(L, (int, float)):
            # Если L - скаляр, все координаты имеют одинаковую вероятность
            L_tensor = torch.full((total_coords,), float(L))
        else:
            # Если L - тензор, преобразуем его в одномерный массив
            L_list = []
            for p in params:
                if torch.is_tensor(L) and L.numel() == p.numel():
                    L_list.append(L.view(-1))
                else:
                    L_list.append(torch.full((p.numel(),), float(L)))
            L_tensor = torch.cat(L_list)
        
        # Вычисляем L_i^alpha
        L_alpha = torch.pow(L_tensor + eps, alpha)
        
        # Нормализуем для получения вероятностей
        probabilities = L_alpha / (L_alpha.sum() + eps)
        
        return probabilities
    
    def _sample_coordinate(self, probabilities):
        """
        Выбирает случайный индекс координаты согласно заданным вероятностям.
        """
        return torch.multinomial(probabilities, 1).item()
    
    def _find_param_and_local_idx(self, params, coord_idx):
        """
        Находит параметр и локальный индекс для глобального индекса координаты.
        
        Returns:
            (param_idx, local_idx): индекс параметра и локальный индекс внутри параметра
        """
        current_idx = 0
        for param_idx, p in enumerate(params):
            param_size = p.numel()
            if current_idx <= coord_idx < current_idx + param_size:
                local_idx = coord_idx - current_idx
                return param_idx, local_idx
            current_idx += param_size
        
        # На случай ошибок округления
        return len(params) - 1, params[-1].numel() - 1
    
    def _get_coordinate_lipschitz(self, params, L, param_idx, local_idx, eps):
        """
        Получает константу Липшица для конкретной координаты.
        """
        if isinstance(L, (int, float)):
            return max(float(L), eps)
        elif torch.is_tensor(L):
            p = params[param_idx]
            if L.numel() == p.numel():
                return max(L.view(-1)[local_idx].item(), eps)
        
        return max(float(L), eps)