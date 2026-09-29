# round6 B1.6 可学习版单元测试：LearnableBonePrior 每骨段高斯先验负对数似然
# 用自定义骨段集 {(0,1)} 隔离验证：NLL = 0.5*((l-μ)/σ)^2 + log σ（l=躯干归一化骨长）；
# 梯度同时流向预测骨长与可训练参数 μ/σ
import math

import torch

from ultralytics.utils.bone_loss import LearnableBonePrior

torch.manual_seed(0)
prior = LearnableBonePrior(bones=[(0, 1)], torso_bones=[(0, 1)])
# 初值：μ=1（归一化骨长 O(1)）、log σ=0（σ=1）


def make_kpts(x1, v=2.0):
    k = torch.zeros(1, 17, 3)
    k[..., 2] = v
    k[0, 0] = torch.tensor([0.0, 0.0, v])
    k[0, 1] = torch.tensor([x1, 0.0, v])
    return k


# 骨长 4=躯干长 4 → l=1.0 == μ → NLL = 0.5*0 + log 1 = 0
k_ok = make_kpts(4.0)
assert prior(k_ok, k_ok).item() < 1e-6

# 骨长 12/4=3.0，μ=1 σ=1 → NLL = 0.5*2^2 + 0 = 2.0（单骨段单样本，分母权重 1）
# 注：l=2 时 log σ 的梯度恰为 0（鞍点），故取 l=3 验证梯度
k_long = make_kpts(12.0, v=2.0).requires_grad_(True)
l = prior(k_long, k_ok)
assert abs(l.item() - 2.0) < 1e-6, l.item()

# 梯度：预测骨长与 μ/σ 参数都收到梯度（可学习先验的意义所在）
l.backward()
assert k_long.grad is not None and k_long.grad.abs().sum() > 0
assert prior.mu.grad is not None and prior.mu.grad.abs().sum() > 0
assert prior.log_sigma.grad is not None and prior.log_sigma.grad.abs().sum() > 0

# σ 项：设 log σ = ln 2（σ=2），l=1.0==μ → NLL = 0.5*0 + ln 2 ≈ 0.6931
prior2 = LearnableBonePrior(bones=[(0, 1)], torso_bones=[(0, 1)])
prior2.log_sigma.data.fill_(math.log(2.0))
l2 = prior2(k_ok, k_ok)
assert abs(l2.item() - math.log(2.0)) < 1e-6, l2.item()

# v==0：两端点任一不可见 → 权重 0 → loss 0（无有效样本时 eps 兜底）
k_v0 = make_kpts(8.0, v=0.0)
assert prior(k_v0, k_v0).item() < 1e-6

print("BONE_PRIOR_OK", l.item(), l2.item())
