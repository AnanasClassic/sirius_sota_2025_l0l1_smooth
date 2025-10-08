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
    """f(x) = 1/n||x||^{2n}"""
    def __init__(self, n: int, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n = n
        self.L0 = 2 * n
        self.L1 = 2 * n - 1

    @property
    def f(self):
        return lambda x: 1/self.n * torch.norm(x, p=2) ** (2 * self.n)

    @property
    def grad(self):
        return lambda x: 2/self.n * torch.norm(x, p=2) ** (2 * (self.n - 1)) * (x)

    @property
    def hess(self):
        def hessian(x):
            norm = torch.norm(x, p=2)
            if self.n == 1:
                return 2 * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
            else:
                term1 = 4 * self.n * (self.n - 1) * norm ** (2 * self.n - 4) * torch.outer(x, x)
                term2 = 2 * self.n * norm ** (2 * self.n - 2) * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
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

    
class SumOfExponents(ExampleFunction):
    """f(x) = sum_i exp(a_i^T x)
    
    Для этой функции:
    grad f(x) = sum_i a_i * exp(a_i^T x)
    ||grad f(x)|| <= sum_i ||a_i|| * exp(a_i^T x)
    
    Hess f(x) = sum_i a_i a_i^T * exp(a_i^T x)
    ||Hess f(x)||_2 <= sum_i ||a_i||^2 * exp(a_i^T x)
    
    По определению (L0, L1)-smoothness:
    ||Hess(x)||_2 <= L0 + L1 * ||grad f(x)||
    
    Для sum of exponentials:
    L0 = 0, L1 = max_i ||a_i||
    """
    def __init__(self, n_dim: int, n_terms: int, scale=1.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.A = torch.randn(n_terms, n_dim) * scale
        # L0 = 0, L1 = max_i ||a_i||
        self.L0 = 0
        self.L1 = torch.norm(self.A, p=2, dim=1).max().item()

    @property
    def parameters(self):
        return [self.A]

    @property
    def f(self):
        return lambda x: torch.exp(self.A @ x).sum()

    @property
    def grad(self):
        return lambda x: (torch.exp(self.A @ x).unsqueeze(1) * self.A).sum(dim=0)

    @property
    def hess(self):
        def hessian(x):
            exp_vals = torch.exp(self.A @ x)
            outer_sum = sum(exp_vals[i] * torch.outer(self.A[i], self.A[i]) for i in range(self.A.shape[0]))
            return outer_sum
        return hessian

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1


class LogSumExp(ExampleFunction):
    """f(x) = log(sum_i exp(a_i^T x))
    
    Для этой функции:
    grad f(x) = sum_i [exp(a_i^T x) / sum_j exp(a_j^T x)] * a_i
              = sum_i p_i(x) * a_i, где p_i(x) - softmax веса
    
    ||grad f(x)|| <= sum_i p_i(x) * ||a_i|| <= max_i ||a_i|| * sum_i p_i(x) = max_i ||a_i||
    
    Hess f(x) = sum_i p_i(x) * a_i a_i^T - grad f(x) * grad f(x)^T
    
    ||Hess f(x)||_2 <= sum_i p_i(x) * ||a_i||^2 <= max_i ||a_i||^2
    
    Поскольку ||grad f(x)|| <= max_i ||a_i||, имеем:
    ||Hess f(x)||_2 <= max_i ||a_i||^2 = max_i ||a_i|| * max_i ||a_i|| <= max_i ||a_i|| * ||grad f(x)||
    
    Таким образом: L0 = 0, L1 = max_i ||a_i||
    """
    def __init__(self, n_dim: int, n_terms: int, scale=1.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.A = torch.randn(n_terms, n_dim) * scale
        self.L0 = 0
        self.L1 = torch.norm(self.A, p=2, dim=1).max().item()

    @property
    def parameters(self):
        return [self.A]

    @property
    def f(self):
        return lambda x: torch.logsumexp(self.A @ x, dim=0)

    @property
    def grad(self):
        def gradient(x):
            inner_products = self.A @ x  # shape: (n_terms,)
            # Softmax weights
            exp_vals = torch.exp(inner_products - torch.max(inner_products))  # numerical stability
            weights = exp_vals / exp_vals.sum()
            # Weighted sum of a_i
            return (weights.unsqueeze(1) * self.A).sum(dim=0)
        return gradient

    @property
    def hess(self):
        def hessian(x):
            inner_products = self.A @ x
            # Softmax weights
            exp_vals = torch.exp(inner_products - torch.max(inner_products))
            weights = exp_vals / exp_vals.sum()
            
            # First term: sum_i p_i(x) * a_i a_i^T
            term1 = sum(weights[i] * torch.outer(self.A[i], self.A[i]) for i in range(self.A.shape[0]))
            
            # Second term: - grad f(x) * grad f(x)^T
            grad_val = (weights.unsqueeze(1) * self.A).sum(dim=0)
            term2 = torch.outer(grad_val, grad_val)
            
            return term1 - term2
        return hessian

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1
    
class Quadratic(ExampleFunction):
    """f(x) = 0.5 <x, Ax> - <b, x> + c
    b = A^-1 x*
    min f(x) = f(x*)"""

    def __init__(self, n_dim, x_star: torch.Tensor, c: float, scale=1.0, seed=None, asym=None):
        super().__init__()
        # det(A@A.t().conj = n_dim! scale ^ n_dim
        # scale = (torch.sqrt(torch.tensor([6.28 * n_dim])) * (n_dim * torch.exp(torch.tensor([-1]))) ** n_dim) ** (-1/n_dim)
        # print(scale)
        if seed is not None:
            torch.manual_seed(seed)
        self.A = torch.zeros(n_dim, n_dim)
        self.A = torch.triu(torch.randn((n_dim, n_dim)))
        self.A *= scale
        if asym is not None:
            self.A[0] *= asym
        self.A = self.A @ self.A.t().conj()
        self.b = self.A @ x_star
        self.c = c

    @property
    def parameters(self):
        return [self.A, self.b, self.c]

    @property
    def f(self):
        return lambda x: 0.5 * (x @ self.A @ x) - (self.b @ x) + self.c

    @property
    def grad(self):
        return lambda x: self.A @ x - self.b

    @property
    def hess(self):
        return lambda x: self.A

    @property
    def l0l1_smooth(self):
        return self.L0, self.L1