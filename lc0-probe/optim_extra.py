"""Optimizers torch lacks (Lion, AdaBound), and make() for picking any optimizer by name."""
import math, torch


class Lion(torch.optim.Optimizer):
    """Chen et al. 2023: the step is the sign of a momentum interpolated with the gradient, decoupled weight decay."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


class AdaBound(torch.optim.Optimizer):
    """Luo et al. 2019: Adam whose per-weight step size is clipped into bounds that close in on final_lr (SGD)."""
    def __init__(self, params, lr=1e-3, final_lr=0.1, betas=(0.9, 0.999), gamma=1e-3, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, final_lr=final_lr, betas=betas, gamma=gamma, eps=eps,
                                      weight_decay=weight_decay))
        self.base_lr = lr

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if not st: st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1; t = st["t"]
                grad = p.grad + g["weight_decay"] * p if g["weight_decay"] else p.grad
                st["m"].mul_(b1).add_(grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                size = g["lr"] * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                final = g["final_lr"] * g["lr"] / self.base_lr
                lo, hi = final * (1 - 1 / (g["gamma"] * t + 1)), final * (1 + 1 / (g["gamma"] * t))
                p.sub_((size / (st["v"].sqrt() + g["eps"])).clamp_(lo, hi) * st["m"])


def make(name, params, lr, wd=0.0):
    """name: adam, adamw, nadam, radam, adamax, adagrad, rmsprop, lion, adabound."""
    extra = {"lion": Lion, "adabound": AdaBound}
    if name in extra: return extra[name](params, lr=lr, weight_decay=wd)
    cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "nadam": torch.optim.NAdam, "radam": torch.optim.RAdam,
           "adamax": torch.optim.Adamax, "adagrad": torch.optim.Adagrad, "rmsprop": torch.optim.RMSprop}[name]
    return cls(params, lr=lr, weight_decay=wd)
