"""Leela Chess Zero style architecture for chess engines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from model.heads import ValueHead, PolicyHead
except ImportError:
    # Fallback for when src is not in path
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from model.heads import ValueHead, PolicyHead


class ResidualBlock(nn.Module):
    """Residual block with batch normalization (Leela Chess Zero style)."""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class LeelaChessNet(nn.Module):
    """
    Leela Chess Zero inspired architecture.
    
    Key features:
    - Deep residual blocks (similar to ResNet)
    - Batch normalization
    - No SE blocks (simpler, faster)
    - Focused on value prediction
    
    Size: ~15-20M parameters
    """
    
    def __init__(self, num_residual_blocks: int = 6, num_moves: int = 1968):
        """
        Initialize Leela Chess Net.
        
        Args:
            num_residual_blocks: Number of residual blocks (default 6)
            num_moves: Number of possible moves for policy head
        """
        super(LeelaChessNet, self).__init__()
        
        # Input: 18 channels
        # Initial convolution
        self.conv_input = nn.Conv2d(18, 256, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Policy head (early exit)
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            board_tensor: Input tensor of shape [batch, 18, 8, 8]
            legal_moves_mask: Optional mask for legal moves
            
        Returns:
            Dictionary with 'value' and 'policy' keys
        """
        # Input convolution
        x = self.relu(self.bn_input(self.conv_input(board_tensor)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy_conv = self.relu(self.policy_bn(self.policy_conv(x)))
        policy_flat = policy_conv.view(policy_conv.size(0), -1)
        policy_logits = self.policy_fc(policy_flat)
        
        # Value head
        value_conv = self.relu(self.value_bn(self.value_conv(x)))
        value_flat = value_conv.view(value_conv.size(0), -1)
        value = self.relu(self.value_fc1(value_flat))
        value = self.tanh(self.value_fc2(value))
        
        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }


class LeelaChessNetLarge(nn.Module):
    """
    Large Leela Chess Zero style architecture.
    
    More residual blocks and wider channels for stronger play.
    Size: ~50-80M parameters
    """
    
    def __init__(self, num_residual_blocks: int = 10, num_moves: int = 1968):
        super(LeelaChessNetLarge, self).__init__()
        
        # Wider initial convolution
        self.conv_input = nn.Conv2d(18, 384, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(384)
        
        # More residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(384) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(384, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)
        
        # Value head
        self.value_conv = nn.Conv2d(384, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 384)
        self.value_fc2 = nn.Linear(384, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.relu(self.bn_input(self.conv_input(board_tensor)))
        
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy_conv = self.relu(self.policy_bn(self.policy_conv(x)))
        policy_flat = policy_conv.view(policy_conv.size(0), -1)
        policy_logits = self.policy_fc(policy_flat)
        
        # Value head
        value_conv = self.relu(self.value_bn(self.value_conv(x)))
        value_flat = value_conv.view(value_conv.size(0), -1)
        value = self.relu(self.value_fc1(value_flat))
        value = self.tanh(self.value_fc2(value))
        
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }

