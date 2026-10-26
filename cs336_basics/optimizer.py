import math

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SGD requires dense gradients")
                state = self.state[parameter]
                step = state.get("step", 0)
                learning_rate = group["lr"] / math.sqrt(step + 1)
                parameter.add_(parameter.grad, alpha=-learning_rate)
                state["step"] = step + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW requires dense gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(gradient, alpha=1 - beta1)
                v.mul_(beta2).add_(gradient * gradient, alpha=1 - beta2)

                learning_rate = group["lr"] * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(m / (v.sqrt() + group["eps"]), alpha=-learning_rate)
        return loss
