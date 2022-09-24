from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups: # dict 
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                if len(state) == 0:
                    state['step'] = 0
                    # exp moving average of moving average (mean) of the gradient - first moment
                    state['exp_mean'] = torch.zeros_like(p.data)
                    # exp moving average of uncentered variance (squared gradient) of the gradient - second moment
                    state['exp_var'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_mean, exp_var = state['exp_mean'], state['exp_var']
                beta1, beta2 = group['betas']

                exp_mean.mul_(beta1).add_(grad, alpha=(1.0-beta1))
                exp_var.mul_(beta2).addcmul_(grad, grad, value=(1.0-beta2))
                denom = exp_var.sqrt().add_(group['eps'])

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group['correct_bias']: # if group['correct_bias'] = True, 

                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']

                    # alpha = alpha * ((bias_correction2) ** (1/2)) / bias_correction1
                    alpha = alpha * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters

                # p.data.addcmul_(-alpha, exp_mean, 1/(torch.sqrt(exp_var) + group['eps']))
                # p.data.addcdiv_(exp_mean, torch.sqrt(exp_var) + group['eps'], value=-alpha)
                p.data.addcdiv_(exp_mean, denom, value=-alpha)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

                if group['weight_decay'] != 0: # w = w - lr * w.grad - lr * wd * w http://www.fast.ai/2018/07/02/adam-weight-decay/
                    # p.data
                    p.data.add_(p.data, alpha=(-alpha * group["weight_decay"]))
                
                # https://towardsdatascience.com/why-adamw-matters-736223f31b5d

        return loss

"""
# param_group example
<class 'generator'>
[{'params': [Parameter containing:
tensor(0.9417, requires_grad=True), Parameter containing:
tensor(0.7757, requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}]

http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
"""