# round6 B1.6 手工版单元测试：BoneStatsLoss 批内统计先验语义
# 参考骨长=本批 fg 样本 GT 骨长（躯干归一化）加权均值；预测与批统计的相对误差平方；
# v==0 两端点任一不可见整段跳过；单样本批退化为逐样本对比（loss≈0）
import torch

from ultralytics.utils.bone_loss import BoneStatsLoss

torch.manual_seed(0)
criterion = BoneStatsLoss()


def make_pose(wrist_x=1.0):
    # 简易站立火柴人（10x10 网格），肩-髋躯干长 4；腕点 9 仅属于骨段 (7,9)
    p = torch.zeros(1, 17, 3)
    p[..., 2] = 2.0
    coords = {
        5: (3, 2),
        6: (7, 2),
        11: (3, 6),
        12: (7, 6),
        7: (2, 4),
        8: (8, 4),
        9: (wrist_x, 6),
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
        p[0, i, 0] = x
        p[0, i, 1] = y
    return p


# 单样本批：μ=该样本自身 GT 骨长，预测==GT → loss 0（退化语义）
single = make_pose()
assert criterion(single, single).item() < 1e-6

# 两样本批：GT 腕点 9 分别为 (1,6)（骨长 √5≈2.236）与 (1.5,5)（骨长 √5/2），
# μ=(√5/4 + √5/8)/2 = 3√5/16；预测两样本都用 (1,6)（骨长 √5/4）
# 相对误差 = (√5/4 − 3√5/16)/(3√5/16) = 1/3，两样本各贡献 (1/3)^2，分母权重和 12×2=24
# 期望 loss = 2×(1/9)/24 = 1/108
gt_a = make_pose(1.0)
gt_b = make_pose(1.5)
gt_b[0, 9, 1] = 5.0
gt_batch = torch.cat([gt_a, gt_b], dim=0)
pred = torch.cat([make_pose(1.0), make_pose(1.0)], dim=0)
l = criterion(pred, gt_batch).item()
assert abs(l - 1 / 108) < 1e-5, l

# v==0：两样本腕点 9 都不可见 → 骨段 (7,9) 无有效样本被跳过，其余骨段预测==GT → loss 0
gt_batch_v0 = gt_batch.clone()
gt_batch_v0[:, 9, 2] = 0.0
pred_far = pred.clone()
pred_far[:, 9, 0] = 100.0
assert criterion(pred_far, gt_batch_v0).item() < 1e-6

# 批统计与逐样本监督的差异：把预测改成「批均值 μ」本身 → 该骨段误差为 0（逐样本监督则非零）
pred_mu = pred.clone()
# 骨段 (7,9) 向量 (x-2, y-4)，μ 向量 = 均值 ((-1,2)+(-0.5,1))/2 = (-0.75,1.5) → 腕点=(1.25,5.5)
pred_mu[0, 9, 0] = 1.25
pred_mu[0, 9, 1] = 5.5
pred_mu[1, 9, 0] = 1.25
pred_mu[1, 9, 1] = 5.5
assert criterion(pred_mu, gt_batch).item() < 1e-6

print("BONE_STATS_OK", l)
