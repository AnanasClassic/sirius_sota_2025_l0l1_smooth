import torch

class ExampleFunction:
    def __init__(self):
        self.L0 = None
        self.L1 = None

    @property
    def parameters(self):
        return None

    @property
    def f(self):
        return None

    @property
    def grad(self):
        return None

    @property
    def hess(self):
        return None

    @property
    def l0l1_smooth(self):
        return None


class NormX2N(ExampleFunction):
    """f(x) = ||x||^{2n}"""
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.L0 = 2 * n
        self.L1 = 2 * n - 1

    @property
    def f(self):
        return lambda x: torch.norm(x, p=2) ** (2 * self.n)

    @property
    def grad(self):
        return lambda x: 2 * self.n * torch.norm(x, p=2) ** (2 * (self.n - 1)) * x

    @property
    def hess(self):
        def hessian(x):
            norm_x = torch.norm(x, p=2)
            if self.n == 1:
                return 2 * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
            else:
                term1 = 4 * self.n * (self.n - 1) * norm_x ** (2 * self.n - 4) * torch.outer(x, x)
                term2 = 2 * self.n * norm_x ** (2 * self.n - 2) * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
                return term1 + term2
        return hessian

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1


class ExponentInnerProduct(ExampleFunction):
    """f(x) = exp(a^T x)"""
    def __init__(self, n_dim: int, scale=1.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.a = torch.randn(n_dim) * scale
        self.L0 = 0
        self.L1 = torch.norm(self.a, p=2).item()

    @property
    def parameters(self):
        return [self.a]

    @property
    def f(self):
        return lambda x: torch.exp(self.a @ x)

    @property
    def grad(self):
        return lambda x: self.a * torch.exp(self.a @ x)

    @property
    def hess(self):
        return lambda x: torch.outer(self.a, self.a) * torch.exp(self.a @ x)

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1


class LogisticFunction(ExampleFunction):
    """f(x) = log(1 + exp(-a^T x))"""
    def __init__(self, n_dim: int, scale=1.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.a = torch.randn(n_dim) * scale
        self.L0 = 0
        self.L1 = torch.norm(self.a, p=2).item()

    @property
    def parameters(self):
        return [self.a]

    @property
    def f(self):
        return lambda x: torch.log(1 + torch.exp(-self.a @ x))

    @property
    def grad(self):
        return lambda x: -self.a / (1 + torch.exp(self.a @ x))

    @property
    def hess(self):
        def hessian(x):
            exp_val = torch.exp(self.a @ x)
            return (exp_val / (1 + exp_val)**2) * torch.outer(self.a, self.a)
        return hessian

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1
