import torch
from torch import nn

# Import the base DSFNet model to be used as the expert
from .DSFNet import DSFNet as Expert

class AMENet(nn.Module):
    """
    AME-Net: A large-scale model integrating multiple independent expert units
    into a single, unified architecture. This design uses modular duplication
    to meet large parameter count requirements and supports dynamic expert selection
    at inference time to balance performance and computational cost.

    Args:
        heads (dict): Dictionary defining the output heads of the network.
        head_conv (int): The number of channels in the final convolution of each head.
        num_experts (int): The total number of expert units to instantiate.
    """
    def __init__(self, heads, head_conv=128, num_experts=60):
        super(AMENet, self).__init__()
        self.num_experts = num_experts
        self.heads = heads

        # Instantiate 60 complete, independent expert models
        # Using nn.ModuleList ensures they are properly registered and saved.
        print(f"🚀 Initializing AME-Net with {self.num_experts} expert units...")
        self.experts = nn.ModuleList(
            [Expert(heads, head_conv) for _ in range(self.num_experts)]
        )
        print("✅ AME-Net initialization complete.")

    def forward(self, x, n_active=1):
        """
        Forward pass with dynamic expert selection.

        Args:
            x (torch.Tensor): The input tensor for the model.
            n_active (int): The number of expert units to activate for this pass.
                              Must be between 1 and self.num_experts.

        Returns:
            A tuple containing the list of output dictionaries and any auxiliary loss.
        """
        # Ensure n_active is within a valid range
        n_active = max(1, min(n_active, self.num_experts))

        # Collect outputs from activated experts
        expert_outputs = []
        for i in range(n_active):
            # Pass input through the i-th expert
            # The base model returns a list [output_dict], so we take the first element
            expert_output_dict = self.experts[i](x)[0]
            expert_outputs.append(expert_output_dict)

        # Aggregate the results by averaging
        # Initialize the final output dictionary with the structure of the first expert's output
        # but with zeroed tensors.
        final_outputs = {head: torch.zeros_like(expert_outputs[0][head]) for head in self.heads}

        # Sum the outputs from all active experts
        for expert_output in expert_outputs:
            for head in self.heads:
                final_outputs[head] += expert_output[head]

        # Average the summed outputs
        for head in self.heads:
            final_outputs[head] /= n_active
        if self.training:
            # 这里的辅助损失设为0.0，因为专家是预训练好的，
            # 如果要训练gating network，这个值需要从gating network返回
            aux_loss = 0.0 
            return [final_outputs], aux_loss
        else:
            return [final_outputs]
        # Return in the expected format: a list containing the output dictionary
        # and a placeholder for auxiliary loss (not used in this architecture).
