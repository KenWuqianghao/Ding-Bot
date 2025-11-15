"""Chess neural network architecture."""
import sys
from pathlib import Path

# CRITICAL: Add src/ to path BEFORE any other imports
src_path = Path(__file__).parent.parent.resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import chess

from model.heads import ValueHead, PolicyHead


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Improved residual block with SE attention."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation and move prediction.
    
    Architecture:
    - Input: [batch, 18, 8, 8] board tensor
    - Body: CNN with residual connections
    - Heads: Value head (evaluation) and Policy head (move prediction)
    
    Default size: ~1.3M parameters (~5MB)
    """
    
    def __init__(self, num_moves: int = 1968):
        """
        Initialize ChessNet.
        
        Args:
            num_moves: Number of possible moves for policy head (default 1968)
        """
        super(ChessNet, self).__init__()
        
        # Input: 18 channels (12 piece channels + 6 metadata)
        # CNN body
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Improved residual block with SE attention
        self.res_block = ResidualBlock(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Heads
        self.value_head = ValueHead(input_dim=64)
        self.policy_head = PolicyHead(input_dim=64, num_moves=num_moves)
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            board_tensor: Input tensor of shape [batch, 18, 8, 8]
            legal_moves_mask: Optional mask tensor of shape [batch, num_moves] for legal moves
            
        Returns:
            Dictionary with 'value' and 'policy' keys
        """
        # CNN body
        x = self.relu(self.bn1(self.conv1(board_tensor)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Residual block with attention
        x = self.res_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        
        # Heads
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            # Mask illegal moves by setting logits to very negative value
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }
    
    def evaluate_position(
        self,
        board_tensor: torch.Tensor,
        legal_moves: List[chess.Move]
    ) -> tuple:
        """
        Evaluate a position and return value and policy probabilities.
        
        Args:
            board_tensor: Input tensor of shape [1, 18, 8, 8]
            legal_moves: List of legal moves
            
        Returns:
            Tuple of (value_score, policy_probs_dict) where policy_probs_dict
            maps moves to probabilities
        """
        self.eval()
        with torch.no_grad():
            # Create legal moves mask (simplified - would need move encoding)
            output = self.forward(board_tensor)
            
            value_score = output['value'].item()
            policy_logits = output['policy'][0]  # [num_moves]
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=0)
            
            # Create dictionary mapping moves to probabilities
            # Note: This is simplified - full implementation would need
            # proper move encoding/decoding
            policy_dict = {}
            # For now, return uniform distribution over legal moves
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                policy_dict[move] = uniform_prob
        
        return value_score, policy_dict


class ChessNetLarge(nn.Module):
    """
    Large neural network for chess position evaluation and move prediction.
    
    Architecture:
    - Input: [batch, 18, 8, 8] board tensor
    - Body: Deeper CNN with multiple residual blocks
    - Heads: Value head (evaluation) and Policy head (move prediction)
    
    Size: ~10-15M parameters (~40-60MB)
    """
    
    def __init__(self, num_moves: int = 1968):
        """
        Initialize ChessNetLarge.
        
        Args:
            num_moves: Number of possible moves for policy head (default 1968)
        """
        super(ChessNetLarge, self).__init__()
        
        # Input: 18 channels
        # Deeper CNN body with more channels
        self.conv1 = nn.Conv2d(18, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        # Multiple improved residual blocks with attention
        self.res_blocks = nn.ModuleList([
            ResidualBlock(512) for _ in range(3)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Larger dense layers
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Heads with larger input
        self.value_head = ValueHead(input_dim=128)
        self.policy_head = PolicyHead(input_dim=128, num_moves=num_moves)
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # CNN body
        x = self.relu(self.bn1(self.conv1(board_tensor)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Multiple residual blocks with attention
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Heads
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }
    
    def evaluate_position(
        self,
        board_tensor: torch.Tensor,
        legal_moves: List[chess.Move]
    ) -> tuple:
        """Evaluate a position and return value and policy probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(board_tensor)
            value_score = output['value'].item()
            policy_logits = output['policy'][0]
            policy_probs = torch.softmax(policy_logits, dim=0)
            
            policy_dict = {}
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                policy_dict[move] = uniform_prob
        
        return value_score, policy_dict


class ChessNetXL(nn.Module):
    """
    Extra-large neural network for maximum chess strength.
    
    Architecture:
    - Input: [batch, 18, 8, 8] board tensor
    - Body: Very deep CNN with many residual blocks
    - Heads: Value head (evaluation) and Policy head (move prediction)
    
    Size: ~50-100M parameters (~200-400MB)
    """
    
    def __init__(self, num_moves: int = 1968):
        """
        Initialize ChessNetXL.
        
        Args:
            num_moves: Number of possible moves for policy head (default 1968)
        """
        super(ChessNetXL, self).__init__()
        
        # Input: 18 channels
        # Very wide and deep CNN body
        self.conv1 = nn.Conv2d(18, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        
        # Many improved residual blocks with attention (6 blocks)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(1024) for _ in range(6)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Very large dense layers
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Heads
        self.value_head = ValueHead(input_dim=256)
        # Larger policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_moves)
        )
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # CNN body
        x = self.relu(self.bn1(self.conv1(board_tensor)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Multiple residual blocks with attention
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Heads
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        
        # Apply legal moves mask if provided
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask.float() - 1.0) * 1e9
        
        return {
            'value': value,
            'policy': policy_logits
        }
    
    def evaluate_position(
        self,
        board_tensor: torch.Tensor,
        legal_moves: List[chess.Move]
    ) -> tuple:
        """Evaluate a position and return value and policy probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(board_tensor)
            value_score = output['value'].item()
            policy_logits = output['policy'][0]
            policy_probs = torch.softmax(policy_logits, dim=0)
            
            policy_dict = {}
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                policy_dict[move] = uniform_prob
        
        return value_score, policy_dict
