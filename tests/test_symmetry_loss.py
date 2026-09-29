# round6 B1.3 单元测试：SymmetryLoss 语义
# 完美对称≈0；肩高差 1（躯干 4）→ (1/4)²/4；v==0 跳过；v==1 半权重
import torch

from ultralytics.utils.bone_loss import SymmetryLoss

torch.manual_seed(0)
criterion = SymmetryLoss()


def make_pose(scale=1.0):
    p = torch.zeros(1, 17, 3)
    p[..., 2] = 2.0
    coords = {
        5: (3, 2), 6: (7, 2), 11: (3, 6), 12: (7, 6),
        7: (2, 4), 8: (8, 4), 9: (1, 6), 10: (9, 6),
        13: (3, 8), 14: (7, 8), 15: (3, 10), 16: (7, 10),
        0: (5, 1), 1: (4, 1), 2: (6, 1), 3: (3.5, 2), 4: (6.5, 2),
    }
    for i, (x, y) in coords.items():
        p[0, i, 0] = x * scale
        p[0, i, 1] = y * scale
    return p


gt = make_pose()
assert criterion(gt, gt).item() < 1e-6

# 右肩抬高 1：dy=1、躯干 4、4 对 → (1/4)²/4 = 0.015625
pred_tilt = make_pose().clone()
pred_tilt[0, 6, 1] = 3.0
l_tilt = criterion(pred_tilt, gt).item()
assert abs(l_tilt - (1 / 4) ** 2 / 4) < 1e-6, l_tilt

# v==0 端点：该对跳过 → 其余全对称 → 0
gt_occ = gt.clone()
gt_occ[0, 6, 2] = 0.0
assert criterion(pred_tilt, gt_occ).item() < 1e-6

# v==1 半权重：0.5×(1/4)²/3.5 = 0.008929
gt_half = gt.clone()
gt_half[0, 6, 2] = 1.0
l_half = criterion(pred_tilt, gt_half).item()
assert abs(l_half - 0.5 * (1 / 4) ** 2 / 3.5) < 1e-6, l_half

print("SYMMETRY_LOSS_OK", l_tilt, l_half)
