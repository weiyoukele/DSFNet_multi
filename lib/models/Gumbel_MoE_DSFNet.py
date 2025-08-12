import torch
from torch import nn
import torch.nn.functional as F
import os

from .DSFNet import DSFNet as DSFNet_expert
from .stNet import load_model


class GumbelGatingNetwork(nn.Module):
    """
    使用Gumbel-Softmax的门控网络。
    - 训练时使用 Gumbel-Softmax 实现可导的近似0/1选择。
    - 评估时使用 ArgMax 实现严格的、确定性的0/1选择。
    """

    def __init__(self, num_experts, top_k=1):
        super(GumbelGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.backbone = nn.Sequential(
            nn.Conv2d(3 * 5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, self.num_experts)

    def forward(self, x):  # 移除了 is_training 参数，直接使用 self.training
        b, c, t, h, w = x.shape
        x_reshaped = x.view(b, c * t, h, w)

        features = self.backbone(x_reshaped)
        features = self.avg_pool(features)
        features = features.view(b, -1)
        logits = self.fc(features)

        # ==================== [核心修改: 区分训练和评估] ====================
        if self.training:
            # 训练时: 使用 Gumbel-Softmax 以保证梯度传播
            gumbel_gates = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
            _, top_k_indices = torch.topk(gumbel_gates, self.top_k, dim=1)

        else:  # 评估时 (model.eval() 模式)
            # 评估时: 使用 ArgMax 做出严格的、确定性的选择
            _, top_k_indices = torch.topk(logits, self.top_k, dim=1)
            gumbel_gates = torch.zeros_like(logits)
            gumbel_gates.scatter_(1, top_k_indices, 1)  # 生成严格的 one-hot 编码
        # =====================================================================

        # 辅助损失 (Load Balancing Loss)
        # 注意：这个损失只在训练时有意义，但在评估时计算它也无妨
        expert_mask = gumbel_gates
        density = expert_mask.mean(dim=0)
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        aux_loss = (density * density_proxy).sum() * self.num_experts

        return gumbel_gates, top_k_indices, aux_loss


class Gumbel_MoE_DSFNet(nn.Module):
    """
    使用Gumbel-Softmax门控进行端到端训练的MoE模型。
    """

    def __init__(self, heads, head_conv=128, num_experts=3, loss_coef=1e-2, pretrained_paths=None):
        super(Gumbel_MoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.loss_coef = loss_coef
        self.heads = heads

        print("🚀 初始化 Gumbel-MoE-DSFNet 模型 (端到端训练)")
        print(f"   - 门控机制: 训练时 Gumbel-Softmax, 评估时 ArgMax")
        print(f"   - 专家总数: {self.num_experts}")

        self.gating_network = GumbelGatingNetwork(self.num_experts)
        self.experts = nn.ModuleList()

        if pretrained_paths is None or len(pretrained_paths) != self.num_experts:
            raise ValueError(f"必须提供一个包含 {self.num_experts} 个预训练模型路径的列表。")

        for i, path in enumerate(pretrained_paths):
            print(f"   - 初始化专家 {i + 1}/{self.num_experts} 从: '{os.path.basename(path)}'")
            expert_model = DSFNet_expert(heads, head_conv)
            if os.path.exists(path):
                expert_model = load_model(expert_model, path)
            else:
                print(f"   - ⚠️ 警告: 路径不存在 {path}。")
            self.experts.append(expert_model)

        print("✅ 所有专家已从预训练权重初始化。")

    def forward(self, x):
        # 1. 获取门控网络的输出
        gumbel_gates, top_k_indices, aux_loss = self.gating_network(x)

        # 2. 计算所有专家的输出
        expert_outputs = [expert(x)[0] for expert in self.experts]

        # 3. 初始化最终输出
        final_outputs = {head: 0.0 for head in self.heads}

        # 4. 使用门控权重进行加权求和
        for i in range(self.num_experts):
            gate_reshaped = gumbel_gates[:, i].view(-1, 1, 1, 1)
            expert_head_dict = expert_outputs[i]
            for head_name, head_tensor in expert_head_dict.items():
                final_outputs[head_name] += gate_reshaped * head_tensor

        return [final_outputs], self.loss_coef * aux_loss