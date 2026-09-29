# round6 B1.4 单元测试：AnglePlausibilityLoss 语义
# 伸直关节无惩罚；折叠过深（<下限）→ (θmin-θ)² 铰链；v==0 跳过；v==1 半权重
import torch

from ultralytics.utils.bone_loss import AnglePlausibilityLoss

torch.manual_seed(0)
criterion = AnglePlausibilityLoss()
D2R = torch.deg2rad(torch.tensor(20.0)).item()


def make_pose():
    p = torch.zeros(1, 17, 3)
    p[..., 2] = 2.0
    coords = {
        5: (3, 2), 6: (7, 2), 11: (3, 6), 12: (7, 6),
        7: (2, 4), 8: (8, 4), 9: (1, 6), 10: (9, 6),
        13: (3, 8), 14: (7, 8), 15: (3, 10), 16: (7, 10),
        0: (5, 1), 1: (4, 1), 2: (6, 1), 3: (3.5, 2), 4: (6.5, 2),
    }
    for i, (x, y) in coords.items():
        p[0, i, 0] = x
        p[0, i, 1] = y
    return p


gt = make_pose()
assert criterion(gt, gt).item() < 1e-6, criterion(gt, gt).item()

# 左膝折叠：踝 15 抬到 (3,6.5) → 膝角 0° → 惩罚 (20°-0°)²/4 个关节
pred_fold = make_pose().clone()
pred_fold[0, 15, 1] = 6.5
l_fold = criterion(pred_fold, gt).item()
assert abs(l_fold - D2R**2 / 4) < 1e-6, l_fold

# v==0 顶点（膝 13）：该三元组跳过 → 其余全合理 → 0
gt_occ = gt.clone()
gt_occ[0, 13, 2] = 0.0
assert criterion(pred_fold, gt_occ).item() < 1e-6

# v==1 半权重：0.5×D2R²/3.5
gt_half = gt.clone()
gt_half[0, 13, 2] = 1.0
l_half = criterion(pred_fold, gt_half).item()
assert abs(l_half - 0.5 * D2R**2 / 3.5) < 1e-6, l_half

print("ANGLE_LOSS_OK", l_fold, l_half)
