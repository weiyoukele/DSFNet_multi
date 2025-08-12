import torch
from torch import nn
import torch.nn.functional as F
import os

# 导入原始的DSFNet作为我们的“专家”
from .DSFNet import DSFNet as DSFNet_expert
from .stNet import load_model  # 我们将使用您项目中的load_model函数


class SparseGatingNetwork(nn.Module):
    """
    稀疏门控网络：为每个专家生成一个分数，并选择分数最高的Top-K个专家。
    这个网络是唯一需要训练的部分。
    """

    def __init__(self, num_experts, top_k=1):
        super(SparseGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 一个轻量级的CNN来提取全局特征
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, self.num_experts)

    def forward(self, x):
        b = x.shape[0]
        features = self.feature_extractor(x)
        features = self.avg_pool(features)
        features = features.view(b, -1)
        logits = self.fc(features)

        top_k_gates, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        sparse_gates = torch.zeros_like(logits)
        # 使用 softmax 对 top_k 的权重进行归一化
        sparse_gates.scatter_(1, top_k_indices, F.softmax(top_k_gates, dim=1))

        # 辅助损失 (Load Balancing Loss)
        expert_mask = torch.zeros_like(logits)
        expert_mask.scatter_(1, top_k_indices, 1)
        density = expert_mask.mean(dim=0)
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        aux_loss = (density * density_proxy).sum() * self.num_experts

        return sparse_gates, top_k_indices, aux_loss


class FrozenExpert_MoE_DSFNet(nn.Module):
    """
    一个拥有冻结专家的MoE模型，只训练门控网络。
    """

    def __init__(self, heads, head_conv=128, num_experts=3, top_k=1, loss_coef=1e-2, pretrained_paths=None):
        super(FrozenExpert_MoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.heads = heads

        print("🚀 初始化 Frozen-Expert-MoE-DSFNet 模型")
        print(f"   - 专家总数: {self.num_experts}")
        print(f"   - 激活专家数 (Top-K): {self.top_k}")

        # 1. 实例化门控网络
        self.gating_network = SparseGatingNetwork(self.num_experts, self.top_k)

        # 2. 实例化、加载并冻结专家
        self.experts = nn.ModuleList()
        if pretrained_paths is None or len(pretrained_paths) != self.num_experts:
            raise ValueError(f"必须提供一个包含 {self.num_experts} 个预训练模型路径的列表。")

        for i, path in enumerate(pretrained_paths):
            print(f"   - 加载并冻结专家 {i + 1}/{self.num_experts} 从: '{os.path.basename(path)}'")
            expert_model = DSFNet_expert(heads, head_conv)

            # 使用您项目中的 load_model 函数加载权重
            if os.path.exists(path):
                expert_model = load_model(expert_model, path)
            else:
                print(f"   - ⚠️ 警告: 路径不存在 {path}。专家 {i + 1} 将使用随机权重！")

            # ==================== [核心步骤: 冻结专家] ====================
            expert_model.eval()  # 设置为评估模式
            for param in expert_model.parameters():
                param.requires_grad = False
            # ==========================================================

            self.experts.append(expert_model)

        print("✅ 所有专家已加载并冻结。")

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. 获取门控输出
        sparse_gates, top_k_indices, aux_loss = self.gating_network(x)

        # 2. 初始化最终输出
        final_outputs = {head: 0.0 for head in self.heads}

        # 3. 遍历批次中的每个样本
        for i in range(batch_size):
            active_indices = top_k_indices[i]
            active_gates = sparse_gates[i][active_indices]
            sample_input = x[i].unsqueeze(0)

            # 4. 遍历被选中的专家
            for k, expert_idx in enumerate(active_indices):
                # 只有被选中的专家会执行前向传播
                with torch.no_grad():  # 再次确认专家不计算梯度
                    expert_output_dict = self.experts[expert_idx](sample_input)[0]

                gate_val = active_gates[k]

                # 5. 加权累加结果
                for head_name, head_tensor in expert_output_dict.items():
                    # 确保 head_tensor 是 [1, C, H, W]
                    if i == 0 and k == 0:
                        # 首次循环时，初始化 final_outputs 张量
                        if not isinstance(final_outputs[head_name], torch.Tensor):
                            final_outputs[head_name] = torch.zeros(batch_size, *head_tensor.shape[1:], device=x.device)

                    final_outputs[head_name][i] += gate_val * head_tensor.squeeze(0)

        return [final_outputs], self.loss_coef * aux_loss