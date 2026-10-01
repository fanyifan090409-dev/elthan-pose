# round6 B1.2 单元测试：BoneConsistencyLoss 语义
# 完美预测≈0；2x 尺度→长度项惩罚；v==0 端点→整段跳过；垂直方向→方向项=1 计入；
# lam 置零的骨段不参与
import torch

from ultralytics.utils.bone_loss import BoneConsistencyLoss

torch.manual_seed(0)
criterion = BoneConsistencyLoss()


def make_pose(scale=1.0):
    # 简易站立火柴人（10x10 网格），肩-髋躯干长 4
    p = torch.zeros(1, 17, 3)
    p[..., 2] = 2.0
    coords = {
        5: (3, 2),
        6: (7, 2),
        11: (3, 6),
        12: (7, 6),
        7: (2, 4),
        8: (8, 4),
        9: (1, 6),
        10: (9, 6),
        13: (3, 8),
        14: (7, 8),
        15: (3, 10),
        16: (7, 10),
        0: (5, 1),
        1: (4, 1),
        2: (6, 1),
        3: (3.5, 2),
        4: (6.5, 2),
    }
    for i, (x, y) in coords.items():
        p[0, i, 0] = x * scale
        p[0, i, 1] = y * scale
    return p


gt = make_pose()
l_perfect = criterion(gt, gt).item()
assert l_perfect < 1e-6, l_perfect

l_2x = criterion(make_pose(2.0), gt).item()
l_15x = criterion(make_pose(1.5), gt).item()
assert 0 < l_15x < l_2x, (l_15x, l_2x)

# v==0 端点：腕关节 (9) 位移巨大但不可见 → 相关骨段 (7,9) 被跳过
pred_bad_wrist = make_pose().clone()
pred_bad_wrist[0, 9, 0] = 100.0
gt_bad_wrist = gt.clone()
gt_bad_wrist[0, 9, 2] = 0.0
assert criterion(pred_bad_wrist, gt_bad_wrist).item() < 1e-6

# 垂直方向：骨段 (7,9) GT 向量 (-1,2)，预测换成同长垂直向量 (-2,-1)（只动腕点 9，它只属于这一根骨段）
# 长度项=||(-1,-3)||^2/4^2=10/16，方向项=1-cos90=1 → (10/16+1)/12 = 0.1354...
pred_perp = make_pose().clone()
pred_perp[0, 9, 0] = 0.0
pred_perp[0, 9, 1] = 3.0
l_perp = criterion(pred_perp, gt).item()
assert abs(l_perp - (10 / 16 + 1) / 12) < 1e-5, l_perp

# v==1 遮挡端点：骨段权重取两端点乘积 1.0×0.5=0.5，归一化分母按加权计数 11.5
# 垂直扰动贡献 1.625 → 0.5×1.625/11.5 = 0.07065...
gt_half = gt.clone()
gt_half[0, 9, 2] = 1.0
l_half_v1 = criterion(pred_perp, gt_half).item()
assert abs(l_half_v1 - 0.5 * (10 / 16 + 1) / 11.5) < 1e-5, l_half_v1

# lam 置零的骨段不参与：把 (7,9) 权重归零后，腕关节位移不再影响 loss
criterion_zero = BoneConsistencyLoss(lam=[0.0 if (i, j) == (7, 9) else 1.0 for i, j in criterion.bones.tolist()])
pred_far = make_pose().clone()
pred_far[0, 9, 0] = 100.0
assert criterion_zero(pred_far, gt).item() < 1e-6

print("BONE_LOSS_OK", l_perfect, l_15x, l_2x, l_perp, l_half_v1)
