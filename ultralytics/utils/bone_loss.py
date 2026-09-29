# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# round6 B1.2：骨长+方向一致性正则（先例 arXiv 2212.02163 骨长约束）
# 躯干长归一化去尺度歧义；两端点任一 v==0 则该骨段不参与，v==1（遮挡）端点按
# B2' 语义半权重（骨段权重取两端点权重乘积）；GT 骨向量退化（长度≈0）时整段跳过；
# 预测骨向量塌缩时方向项无定义只留长度项。
import torch
from torch import nn

BONES_COCO17 = [
    (5, 7), (7, 9),  # 左肩-左肘、左肘-左腕
    (6, 8), (8, 10),  # 右肩-右肘、右肘-右腕
    (11, 13), (13, 15),  # 左髋-左膝、左膝-左踝
    (12, 14), (14, 16),  # 右髋-右膝、右膝-右踝
    (5, 11), (6, 12),  # 左肩-左髋、右肩-右髋
    (5, 6), (11, 12),  # 双肩连线、双髋连线
]
TORSO_BONES_COCO17 = [(5, 11), (6, 12)]


class BoneConsistencyLoss(nn.Module):
    """骨段向量一致性损失：mean_e [ ‖ê−e‖² + (1−cos θ_e) ]，ê=预测骨向量、e=GT 骨向量，均除以 GT 躯干长。"""

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
            (torch.Tensor): Scalar loss, normalized by valid bone count.
        """
        p = pred_kpts[..., :2]
        g = gt_kpts[..., :2]
        v = (gt_kpts[..., 2] > 0).float()  # 只看 GT 可见性，v==0 端点的骨段不监督
        # B2' 语义一致：v==1（遮挡）端点半权重，骨段权重取两端点权重乘积
        w_v = torch.where(gt_kpts[..., 2] == 1, 0.5, v)

        bone_mask = w_v[:, self.bones[:, 0]] * w_v[:, self.bones[:, 1]]  # (N,E) 两端点权重乘积
        pv = p[:, self.bones[:, 1]] - p[:, self.bones[:, 0]]  # (N,E,2) 预测骨向量
        gv = g[:, self.bones[:, 1]] - g[:, self.bones[:, 0]]  # GT 骨向量

        # 躯干长：肩-髋两段均值（GT 侧），预测与 GT 共用同一 T，躯干无标注时退化为 1（不归一）
        torso_v = (g[:, self.torso[:, 1]] - g[:, self.torso[:, 0]]).norm(dim=-1)  # (N,T)
        torso_vis = w_v[:, self.torso[:, 0]] * w_v[:, self.torso[:, 1]]
        torso_len = (torso_v * torso_vis).sum(-1).add(self.eps) / torso_vis.sum(-1).add(self.eps)

        e_len = (pv - gv).pow(2).sum(-1) / torso_len.unsqueeze(-1).pow(2).add(self.eps)  # 长度项（T 归一化）

        p_norm = pv.norm(dim=-1)
        g_norm = gv.norm(dim=-1)
        cos = (pv * gv).sum(-1) / (p_norm * g_norm).add(self.eps)
        e_dir = (1.0 - cos).clamp(min=0.0)
        e_dir = torch.where(p_norm > self.eps, e_dir, torch.zeros_like(e_dir))  # 预测塌缩：方向无定义只留长度项

        valid = bone_mask * (g_norm > self.eps).float()  # GT 骨向量退化整段跳过
        total = (e_len + e_dir) * self.lam.view(1, -1) * valid
        return total.sum() / valid.sum().add(self.eps)
