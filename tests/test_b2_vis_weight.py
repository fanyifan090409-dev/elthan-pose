# round6 B2' 单元测试：软权重 mask 的 KeypointLoss 语义
# v==1（遮挡）→ 0.5 权重；v==0 → 0 权重；归一化因子按「非零即算标注」计数
import torch

from ultralytics.utils.loss import KeypointLoss

torch.manual_seed(0)
kpl = KeypointLoss(sigmas=torch.tensor([0.5, 0.5]))

# 两个关键点误差相同（d=1），面积相同
pred = torch.zeros(1, 2, 2)
gt = torch.zeros(1, 2, 3)
gt[..., 0] = 1.0
area = torch.full((1, 1), 10.0)

mask_full = torch.tensor([[1.0, 1.0]])  # 两个 v==2
mask_half = torch.tensor([[1.0, 0.5]])  # kpt1 v==1 → 半权重
mask_zero = torch.tensor([[1.0, 0.0]])  # kpt1 v==0 → 不监督

l_full = kpl(pred, gt, mask_full, area).item()
l_half = kpl(pred, gt, mask_half, area).item()
l_zero = kpl(pred, gt, mask_zero, area).item()

# v==1 半权重：总损失 = factor(1) × (t + 0.5t)/2 = 0.75 × full
assert abs(l_half / l_full - 0.75) < 1e-6, (l_half, l_full)
# v==0 不监督：只剩 kpt0，归一化因子 2/1=2 → 损失回到 full 水平（均值语义，非除错）
assert abs(l_zero / l_full - 1.0) < 1e-6, (l_zero, l_full)
print(f"B2_WEIGHT_OK full={l_full:.6f} half={l_half:.6f} zero={l_zero:.6f}")
