"""Tiny Leela Chess Zero style architecture for <10MB models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
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


class LeelaChessNetTiny(nn.Module):
    """
    Tiny Leela Chess Zero style architecture optimized for <10MB.
    
    Architecture:
    - Reduced channels: 96 (instead of 128) - FURTHER REDUCED for <10MB
    - Fewer residual blocks: 3 (instead of 4) - REDUCED for <10MB
    - Smaller value head: 64 -> 32 -> 1 - REDUCED for <10MB
    - Optimized for FP16/INT8 quantization
    
    Target size: <10MB (FP16)
    """
    
    def __init__(self, num_residual_blocks: int = 3, channels: int = 96, num_moves: int = 1968):
        """
        Initialize Tiny Leela Chess Net.
        
        Args:
            num_residual_blocks: Number of residual blocks (default 3, reduced from 4)
            channels: Number of channels (default 96, reduced from 128)
            num_moves: Number of possible moves for policy head
        """
        super(LeelaChessNetTiny, self).__init__()
        
        # Input: 18 channels
        # Reduced initial convolution (96 channels instead of 128)
        self.conv_input = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower (fewer blocks)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        
        # Policy head (simplified)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)
        
        # Value head (smaller - further reduced)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)   # Reduced from 128
        self.value_fc2 = nn.Linear(64, 32)       # Reduced from 64
        self.value_fc3 = nn.Linear(32, 1)
        
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
        value = self.relu(self.value_fc2(value))
        value = self.tanh(self.value_fc3(value))
        
        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }

