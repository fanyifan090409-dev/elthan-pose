# round6 B1.5 单元测试：uncertainty_weights 解析解自动配权（arXiv 2408.07985）
# ω_j = softmax_j((1/L_j)/T)；输入须 detach（stopgradient）；L+eps 防除零；
# 温度越大越均匀；单元素恒 1.0
import math

import torch

from ultralytics.utils.bone_loss import uncertainty_weights

e = math.e

# L=[1,2], T=1 → ω ∝ [exp(1), exp(0.5)]
w = uncertainty_weights([torch.tensor(1.0), torch.tensor(2.0)], 1.0)
ex = [e**1.0, e**0.5]
ex = [x / sum(ex) for x in ex]
assert torch.allclose(w, torch.tensor(ex), atol=1e-6), (w, ex)
assert abs(w.sum().item() - 1.0) < 1e-6

# 预测完美 L=0：1/(0+eps) 巨大但有限，权重几乎全归该项，不出现 inf/nan
w_zero = uncertainty_weights([torch.tensor(0.0), torch.tensor(1.0)], 1.0)
assert torch.isfinite(w_zero).all()
assert w_zero[0] > 0.999999 and w_zero[1] < 1e-6, w_zero

# 温度越低越尖锐
w_hot = uncertainty_weights([torch.tensor(1.0), torch.tensor(2.0)], 0.1)
w_cold = uncertainty_weights([torch.tensor(1.0), torch.tensor(2.0)], 10.0)
assert w_hot[0] > w_cold[0], (w_hot, w_cold)

# stopgradient 契约：调用方传 detach 后的损失，输出无梯度通路
v = torch.tensor(1.0, requires_grad=True)
w_det = uncertainty_weights([v.detach(), torch.tensor(2.0)], 1.0)
assert not w_det.requires_grad

# 单元素退化为 1.0（softmax 单值恒 1）
w_one = uncertainty_weights([torch.tensor(3.7)], 1.0)
assert torch.allclose(w_one, torch.tensor([1.0])), w_one

print("UNCERTAINTY_LOSS_OK", w.tolist(), w_zero.tolist(), w_hot.tolist(), w_cold.tolist())
