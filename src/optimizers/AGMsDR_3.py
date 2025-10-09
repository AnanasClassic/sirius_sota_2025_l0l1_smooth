import numpy as np
import torch
from tqdm import tqdm

class AGMsDR_3:
    def __init__(self, x0, f, T, grad_f, L0, L1, dtype=torch.float32):
        #self.viz = OptimizerVisualizer(window_size=1000)
        self.dtype = dtype
        self.golden_ratio = torch.tensor((np.sqrt(5) + 1) / 2, dtype=dtype)
        self.x_k = x0.clone()
        self.v_k = x0.clone()
        self.f = f
        self.L0 = L0
        self.L1 = L1
        self.grad_f = grad_f
        self.T = T
        self.A_k = 0
        self.device = torch.device('cpu')
        self.history = []
        self.eps = 1e-6
        self.dim = len(x0)
        self.zeta = lambda x: 1/2 * (x - self.v_k)
        self.beta_opt = self.golden_section_search(
                lambda beta: self.f(self.v_k + beta * (self.x_k - self.v_k))
            )
        self.y_k = self.v_k + self.beta_opt * (self.x_k - self.v_k)

    def golden_section_search(self, f, a=0, b=1, tol=1e-5):
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        b = torch.as_tensor(b, dtype=self.dtype, device=self.device)
        d = a + (b - a) / self.golden_ratio
        c = b - (b - a) / self.golden_ratio
        if d < c:
            c, d = d, c
        while abs(b - a) > tol:
            if f(c) > f(d):
                a = d
            else:
                b = c
            d = a + (b - a) / self.golden_ratio
            c = b - (b - a) / self.golden_ratio
            if d < c:
                c, d = d, c
        return (b + a) / 2

    def AGMsDR(self, number_iter=100000, seed=None, verbose=False):
        a_k = [0.0] * self.dim  # ИЗМЕНЕНО ранее: используем float, чтобы избежать целочисленной арифметики
        self.A_k = [0.0] * self.dim  # ИЗМЕНЕНО ранее: то же для аккумулятора A_k

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        for i in tqdm(range(number_iter), disable=not verbose):
            j = np.random.randint(0, self.dim)

            # Линейный поиск по координате j: меняем только одну координату, но считаем f на полном векторе
            def phi(beta):
                x_beta = self.v_k.clone()  # ИЗМЕНЕНО ранее: строим полный вектор для корректного вызова f
                x_beta[j] = self.v_k[j] + beta * (
                            self.x_k[j] - self.v_k[j])  # ИЗМЕНЕНО ранее: обновляем только j-ю координату
                return self.f(
                    x_beta).item()  # ИЗМЕНЕНО ранее: возвращаем float, чтобы golden_section_search мог делать сравнения

            self.beta_opt = self.golden_section_search(phi)  # ИЗМЕНЕНО ранее: передаём обёртку phi, а не f от скаляра

            self.y_k = self.v_k.clone()  # ИЗМЕНЕНО ранее: работаем с полным вектором y_k
            self.y_k[j] = self.v_k[j] + self.beta_opt * (
                        self.x_k[j] - self.v_k[j])  # ИЗМЕНЕНО ранее: обновляем только j-ю координату

            x_tmp = self.T(self.y_k, self.f, self.grad_f, 0, self.L0,
                           self.L1)  # ИЗМЕНЕНО: передаём в T полный вектор y_k (T/grad_f ожидают вектор)
            self.x_k[j] = x_tmp[j]  # ИЗМЕНЕНО: берём только j-ю координату результата T (координатный апдейт)

            denominator = max(1e-12, 2.0 * (self.f(self.y_k) - self.f(
                self.x_k)).item())  # ИЗМЕНЕНО ранее: считаем f на полных векторах + .item() в float

            g_j = self.grad_f(self.y_k)[j]  # ИЗМЕНЕНО ранее: берём только j-ю компоненту градиента
            g_j2 = float(g_j) * float(g_j)  # ИЗМЕНЕНО ранее: квадрат j-й компоненты градиента как float
            M_k = max(self.eps, g_j2 / denominator)  # ИЗМЕНЕНО ранее: используем только j-ю компоненту в оценке M_k

            a_k[j] = (1.0 + np.sqrt(1.0 + 4.0 * self.A_k[j] * M_k)) / (
                        2.0 * M_k)  # ИЗМЕНЕНО ранее: явные float-константы
            self.A_k[j] = self.A_k[j] + a_k[j]  # ИЗМЕНЕНО ранее

            self.v_k[j] = self.v_k[j] - a_k[j] * g_j  # ИЗМЕНЕНО ранее: покоординатное обновление
            self.history.append(self.f(self.x_k).item())  # ИЗМЕНЕНО ранее: сохраняем float вместо 0D-тензора

        return self.history
