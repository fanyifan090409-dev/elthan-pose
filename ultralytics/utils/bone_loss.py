# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# round6 解剖先验损失（B1.2 骨长+方向 / B1.3 对称性 / B1.4 角度合理性）
# 共用语义：端点可见性权重取乘积（v==0 不监督、v==1 半权重，与 B2' 一致）；
# 尺度用 GT 躯干长（肩-髋均值）归一化；GT 向量退化时跳过对应项。
import torch
from torch import nn

BONES_COCO17 = [
    (5, 7),
    (7, 9),  # 左肩-左肘、左肘-左腕
    (6, 8),
    (8, 10),  # 右肩-右肘、右肘-右腕
    (11, 13),
    (13, 15),  # 左髋-左膝、左膝-左踝
    (12, 14),
    (14, 16),  # 右髋-右膝、右膝-右踝
    (5, 11),
    (6, 12),  # 左肩-左髋、右肩-右髋
    (5, 6),
    (11, 12),  # 双肩连线、双髋连线
]
TORSO_BONES_COCO17 = [(5, 11), (6, 12)]
SYMMETRY_PAIRS_COCO17 = [(5, 6), (11, 12), (13, 14), (15, 16)]  # 左右肩/髋/膝/踝
# (顶点, 端点1, 端点2, 折叠角下限/度)：肘 10°、膝 20°（内角低于下限视为解剖不合理）
ANGLE_TRIPLES_COCO17 = [
    (7, 5, 9, 10.0),  # 左肘
    (8, 6, 10, 10.0),  # 右肘
    (13, 11, 15, 20.0),  # 左膝
    (14, 12, 16, 20.0),  # 右膝
]


def vis_weights(gt_kpts: torch.Tensor) -> torch.Tensor:
    """GT 可见性软权重：v==2→1.0、v==1（遮挡）→0.5、v==0→0（B2' 语义）。."""
    return torch.where(gt_kpts[..., 2] == 1, 0.5, (gt_kpts[..., 2] > 0).float())


def gt_torso_len(g: torch.Tensor, w_v: torch.Tensor, torso_bones: torch.Tensor, eps: float) -> torch.Tensor:
    """GT 躯干长（肩-髋两段加权均值）；躯干无标注时退化为 1（不归一）。."""
    torso_v = (g[:, torso_bones[:, 1]] - g[:, torso_bones[:, 0]]).norm(dim=-1)  # (N,T)
    torso_vis = w_v[:, torso_bones[:, 0]] * w_v[:, torso_bones[:, 1]]
    return (torso_v * torso_vis).sum(-1).add(eps) / torso_vis.sum(-1).add(eps)


class BoneConsistencyLoss(nn.Module):
    """骨段向量一致性损失：mean_e [ ‖ê−e‖² + (1−cos θ_e) ]，ê=预测骨向量、e=GT 骨向量，均除以 GT 躯干长。."""

    def __init__(self, bones=BONES_COCO17, torso_bones=TORSO_BONES_COCO17, lam=None, eps=1e-7):
        """Initialize with bone edge pairs, torso normalization bones and per-bone weights."""
        super().__init__()
        self.register_buffer("bones", torch.tensor(bones, dtype=torch.long))  # (E,2) 方向 i→j（近端→远端）
        self.register_buffer("torso", torch.tensor(torso_bones, dtype=torch.long))  # (T,2) 躯干参照段
        lam = torch.ones(len(bones)) if lam is None else torch.tensor(lam, dtype=torch.float32)
        self.register_buffer("lam", lam)  # 每骨段权重，默认均匀
        self.eps = eps

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor) -> torch.Tensor:
        """Compute bone consistency loss for aligned predictions and ground truth.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints (N, 17, 3).
            gt_kpts (torch.Tensor): Ground truth keypoints (N, 17, 3), same coordinate space as pred.

        Returns:
            (torch.Tensor): Scalar loss, normalized by weighted valid bone count.
        """
        p = pred_kpts[..., :2]
        g = gt_kpts[..., :2]
        w_v = vis_weights(gt_kpts)

        bone_mask = w_v[:, self.bones[:, 0]] * w_v[:, self.bones[:, 1]]  # (N,E) 两端点权重乘积
        pv = p[:, self.bones[:, 1]] - p[:, self.bones[:, 0]]  # (N,E,2) 预测骨向量
        gv = g[:, self.bones[:, 1]] - g[:, self.bones[:, 0]]  # GT 骨向量

        torso_len = gt_torso_len(g, w_v, self.torso, self.eps)

        e_len = (pv - gv).pow(2).sum(-1) / torso_len.unsqueeze(-1).pow(2).add(self.eps)  # 长度项（T 归一化）

        p_norm = pv.norm(dim=-1)
        g_norm = gv.norm(dim=-1)
        cos = (pv * gv).sum(-1) / (p_norm * g_norm).add(self.eps)
        e_dir = (1.0 - cos).clamp(min=0.0)
        e_dir = torch.where(p_norm > self.eps, e_dir, torch.zeros_like(e_dir))  # 预测塌缩：方向无定义只留长度项

        valid = bone_mask * (g_norm > self.eps).float()  # GT 骨向量退化整段跳过
        total = (e_len + e_dir) * self.lam.view(1, -1) * valid
        return total.sum() / valid.sum().add(self.eps)


class SymmetryLoss(nn.Module):
    """左右对称软惩罚：配对关节 y 坐标差（躯干归一化）的 L2。."""

    def __init__(self, pairs=SYMMETRY_PAIRS_COCO17, torso_bones=TORSO_BONES_COCO17, eps=1e-7):
        super().__init__()
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))  # (P,2)
        self.register_buffer("torso", torch.tensor(torso_bones, dtype=torch.long))
        self.eps = eps

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor) -> torch.Tensor:
        """Pairwise y-difference penalty, torso-normalized and weighted by endpoint visibility product."""
        p = pred_kpts[..., :2]
        g = gt_kpts[..., :2]
        w_v = vis_weights(gt_kpts)

        w = w_v[:, self.pairs[:, 0]] * w_v[:, self.pairs[:, 1]]  # (N,P)
        torso_len = gt_torso_len(g, w_v, self.torso, self.eps)

        dy = (p[:, self.pairs[:, 0], 1] - p[:, self.pairs[:, 1], 1]).abs()  # (N,P)
        e = dy.div(torso_len.unsqueeze(-1).add(self.eps)).pow(2) * w
        return e.sum() / w.sum().add(self.eps)


class AnglePlausibilityLoss(nn.Module):
    """关节角度合理性：三点内角低于解剖下限（肘 10°/膝 20°）时软惩罚，权重取三点可见性乘积。.

    只惩罚「折叠过深」一侧——内角上界 180° 是站立常态，不做上界惩罚（超伸在 2D 内角中 不可表示）；域特定高斯先验留待产品数据集训练时另做，此处保持保守弱正则。
    """

    def __init__(self, triples=ANGLE_TRIPLES_COCO17, eps=1e-7):
        super().__init__()
        # (J,4): 顶点, 端点1, 端点2, 折叠角下限（度）
        self.register_buffer("triples", torch.tensor(triples, dtype=torch.float32))
        self.eps = eps

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor) -> torch.Tensor:
        """Hinge penalty for joints flexed below the anatomical minimum angle."""
        p = pred_kpts[..., :2]
        w_v = vis_weights(gt_kpts)

        v = self.triples[:, :3].to(torch.long)  # (J,3)
        w = w_v[:, v[:, 0]] * w_v[:, v[:, 1]] * w_v[:, v[:, 2]]  # (N,J) 三点权重乘积
        a = p[:, v[:, 1]] - p[:, v[:, 0]]  # 端点1→顶点
        b = p[:, v[:, 2]] - p[:, v[:, 0]]  # 端点2→顶点
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).add(self.eps)
        theta = cos.clamp(-1.0, 1.0).acos()  # 内角（弧度）
        theta_min = self.triples[:, 3].deg2rad().unsqueeze(0)
        e = (theta_min - theta).clamp(min=0.0).pow(2) * w
        return e.sum() / w.sum().add(self.eps)


def uncertainty_weights(losses: list, temperature: float) -> torch.Tensor:
    """B1.5 解剖先验自动配权（arXiv 2408.07985 解析解 σ*=√L 的 softmax 形式）。.

    ω_j = softmax_j((1/L_j)/T)。调用方须传入 detach 后的损失（stopgradient：权重只做 平衡，不给损失本身回传梯度）；L_j + eps 防除零（预测完美时 L→0，权重全归该项）。
    温度越大分布越均匀；单元素退化为 1.0（softmax 单值恒 1）。
    """
    inv = torch.stack([1.0 / (v + 1e-8) for v in losses])
    return (inv / temperature).softmax(dim=0)


class BoneStatsLoss(nn.Module):
    """B1.6 手工版对照：批内统计先验——参考骨长为本批 fg 样本 GT 骨长（躯干归一化）的加权均值。.

    与 BoneConsistencyLoss（逐样本 GT 监督）不同：监督信号是批统计量 μ_j 而非单个样本的 GT 向量；单样本批退化为逐样本对比。骨段两端任一 v==0 或 GT 骨向量退化则整段跳过，
    本批无有效样本的骨段不参与。
    """

    def __init__(self, bones=BONES_COCO17, torso_bones=TORSO_BONES_COCO17, eps=1e-7):
        super().__init__()
        self.register_buffer("bones", torch.tensor(bones, dtype=torch.long))
        self.register_buffer("torso_bones", torch.tensor(torso_bones, dtype=torch.long))
        self.eps = eps

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor) -> torch.Tensor:
        p = pred_kpts[..., :2]
        g = gt_kpts[..., :2]
        w_v = vis_weights(gt_kpts)
        torso = gt_torso_len(g, w_v, self.torso_bones, self.eps)  # (N,)
        i, j = self.bones[:, 0], self.bones[:, 1]
        g_len = (g[:, j] - g[:, i]).norm(dim=-1) / torso.unsqueeze(-1)  # (N,J)
        p_len = (p[:, j] - p[:, i]).norm(dim=-1) / torso.unsqueeze(-1)  # (N,J)
        w = w_v[:, i] * w_v[:, j] * (g_len > self.eps)  # 退化 GT 骨向量不进入统计
        wsum = w.sum(0)  # (J,)
        mu = (g_len * w).sum(0) / wsum.add(self.eps)  # 批内统计参考 (J,)
        valid = (wsum > self.eps) & (mu > self.eps)
        e = ((p_len - mu.unsqueeze(0)) / mu.unsqueeze(0).add(self.eps)).pow(2) * w  # 相对误差平方
        return e[:, valid].sum() / w[:, valid].sum().add(self.eps)


class LearnableBonePrior(nn.Module):
    """B1.6 可学习版：每骨段可训练 (μ,σ) 高斯先验，损失为负对数似然（躯干归一化骨长）。.

    μ 初值 1.0（归一化骨长 O(1)）、log σ 初值 0（σ=1）；梯度同时流向预测骨长与先验参数。 GT 只用于可见性掩码与躯干归一化，不提供逐样本长度监督（与 BoneStatsLoss 批统计对照）。
    """

    def __init__(self, bones=BONES_COCO17, torso_bones=TORSO_BONES_COCO17, eps=1e-7):
        super().__init__()
        self.register_buffer("bones", torch.tensor(bones, dtype=torch.long))
        self.register_buffer("torso_bones", torch.tensor(torso_bones, dtype=torch.long))
        self.eps = eps
        n = len(bones)
        self.mu = nn.Parameter(torch.ones(n))
        self.log_sigma = nn.Parameter(torch.zeros(n))

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor) -> torch.Tensor:
        p = pred_kpts[..., :2]
        w_v = vis_weights(gt_kpts)
        torso = gt_torso_len(gt_kpts[..., :2], w_v, self.torso_bones, self.eps)  # (N,)
        i, j = self.bones[:, 0], self.bones[:, 1]
        p_len = (p[:, j] - p[:, i]).norm(dim=-1) / torso.unsqueeze(-1)  # (N,J)
        w = w_v[:, i] * w_v[:, j]  # (N,J)
        sigma = self.log_sigma.exp()
        nll = 0.5 * ((p_len - self.mu.unsqueeze(0)) / sigma.unsqueeze(0)).pow(2) + self.log_sigma.unsqueeze(0)
        return (nll * w).sum() / w.sum().add(self.eps)
