# round6 冒烟：真实训练路径跑一个合成 batch（含 v=0/1/2 混合），
# 逐档验证 B2' 软加权与 B1.2-B1.6 解剖先验开关：全关键集合恰好 5 项，逐项开启键齐备、反向全通
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT

torch.manual_seed(0)
model = YOLO("/home/ubuntu/isef/runs/pose-round4/weights/best.pt").model
model.train()
# best.pt 经 strip_optimizer 保存，参数 requires_grad 被关掉（torch.load 会保留该标志）。
# 真实训练走「新建模型 + load_state_dict」，参数默认开 grad；冒烟里手动恢复，否则反向无梯度可流
model.requires_grad_(True)
# 训练时 PoseTrainer 会把 args 换成含 hyp 的命名空间（box/pose/kobj 等），冒烟里手动注入
model.args = get_cfg(DEFAULT_CFG_DICT)

bs = 2
kpts = torch.tensor(
    [
        # 样本 0：v 混 0/1/2（遮挡场景）
        [
            [0.30, 0.20, 2],
            [0.32, 0.15, 2],
            [0.34, 0.12, 1],
            [0.36, 0.15, 2],
            [0.38, 0.20, 2],
            [0.35, 0.22, 2],
            [0.40, 0.21, 1],
            [0.42, 0.24, 2],
            [0.44, 0.27, 2],
            [0.46, 0.29, 2],
            [0.40, 0.50, 2],
            [0.42, 0.52, 1],
            [0.43, 0.70, 2],
            [0.45, 0.72, 2],
            [0.46, 0.74, 0],
            [0.35, 0.12, 0],
            [0.40, 0.12, 2],
        ],
        # 样本 1：v 混 0/1/2
        [
            [0.50, 0.20, 2],
            [0.52, 0.15, 2],
            [0.54, 0.12, 2],
            [0.56, 0.15, 1],
            [0.58, 0.20, 2],
            [0.55, 0.22, 2],
            [0.60, 0.21, 2],
            [0.62, 0.24, 2],
            [0.64, 0.27, 1],
            [0.66, 0.29, 2],
            [0.60, 0.50, 2],
            [0.62, 0.52, 2],
            [0.63, 0.70, 2],
            [0.65, 0.72, 2],
            [0.66, 0.74, 2],
            [0.55, 0.12, 0],
            [0.60, 0.12, 2],
        ],
    ],
    dtype=torch.float32,
)
batch = {
    "img": torch.rand(bs, 3, 640, 640),  # 模拟 preprocess 后的归一化图
    "batch_idx": torch.tensor([[0], [1]]),
    "cls": torch.tensor([[0], [0]], dtype=torch.float32),
    "bboxes": torch.tensor([[0.1, 0.1, 0.7, 0.9], [0.2, 0.15, 0.8, 0.95]]),
    "keypoints": kpts,
}

loss, loss_items = model(batch)
print("loss_sum=", loss.sum().item())
print("loss_items=", {k: round(v.item(), 4) for k, v in loss_items.items()})
assert torch.isfinite(loss.sum()), "loss is not finite"
# 全关时键集合必须只有原始 5 项（解剖先验默认不出现，避免污染基线）
assert set(loss_items) == {"box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"}, "unexpected keys when all off"
loss.sum().backward()
print("SMOKE_OK (bone off)")

# B1.2 开启：hyp.bone 权重注入后 bone_loss 键出现且反向仍通
model.args.bone = 1.0
loss2, loss_items2 = model(batch)
print("loss_items(bone on)=", {k: round(v.item(), 4) for k, v in loss_items2.items()})
assert "bone_loss" in loss_items2, "bone_loss key missing when hyp.bone > 0"
assert torch.isfinite(loss2.sum()), "loss is not finite (bone on)"
loss2.sum().backward()
print("SMOKE_OK (bone on)")

# B1.3 开启：bone+symmetry 同时开，键齐备且反向仍通
model.args.symmetry = 1.0
loss3, loss_items3 = model(batch)
print("loss_items(bone+sym on)=", {k: round(v.item(), 4) for k, v in loss_items3.items()})
assert {"bone_loss", "symmetry_loss"} <= set(loss_items3), "anatomy loss keys missing"
assert torch.isfinite(loss3.sum()), "loss is not finite (bone+sym on)"
loss3.sum().backward()
print("SMOKE_OK (bone+sym on)")

# B1.4 开启：三项解剖先验全开
# 注意键名是 angle_prior：default.yaml 的 angle 已被 OBB 定向角增益占用（默认 1.0），不可用作开关
model.args.angle_prior = 1.0
loss4, loss_items4 = model(batch)
print("loss_items(all on)=", {k: round(v.item(), 4) for k, v in loss_items4.items()})
assert {"bone_loss", "symmetry_loss", "angle_prior_loss"} <= set(loss_items4), "anatomy loss keys missing"
assert torch.isfinite(loss4.sum()), "loss is not finite (all on)"
loss4.sum().backward()
print("SMOKE_OK (all anatomy on)")

# B1.5 开启：三项全开 + 自动配权（ω=softmax((1/L)/T)，stopgradient；手工权重退化为开门）
model.args.anatomy_uncertainty = 1.0
loss5, loss_items5 = model(batch)
print("loss_items(uncertainty on)=", {k: round(v.item(), 4) for k, v in loss_items5.items()})
assert {"bone_loss", "symmetry_loss", "angle_prior_loss"} <= set(loss_items5), "anatomy loss keys missing"
assert torch.isfinite(loss5.sum()), "loss is not finite (uncertainty on)"
loss5.sum().backward()
print("SMOKE_OK (uncertainty weighting on)")

# B1.6 对照（手工版批统计先验）：关掉自动配权，单独开 bone_stats
model.args.anatomy_uncertainty = 0.0
model.args.bone_stats = 1.0
loss6, loss_items6 = model(batch)
print("loss_items(bone_stats on)=", {k: round(v.item(), 4) for k, v in loss_items6.items()})
assert "bone_stats_loss" in loss_items6, "bone_stats_loss key missing"
assert torch.isfinite(loss6.sum()), "loss is not finite (bone_stats on)"
loss6.sum().backward()
print("SMOKE_OK (bone_stats on)")

# B1.6 对照（可学习版高斯先验）：单独开 bone_prior，可训练 μ/σ 参数进梯度
model.args.bone_prior = 1.0
loss7, loss_items7 = model(batch)
print("loss_items(bone_prior on)=", {k: round(v.item(), 4) for k, v in loss_items7.items()})
assert "bone_prior_loss" in loss_items7, "bone_prior_loss key missing"
assert torch.isfinite(loss7.sum()), "loss is not finite (bone_prior on)"
loss7.sum().backward()
print("SMOKE_OK (bone_prior on)")
