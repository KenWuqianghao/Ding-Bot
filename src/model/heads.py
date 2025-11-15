"""Neural network heads for value and policy prediction."""
import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Value head for position evaluation.
    
    Outputs a scalar evaluation score.
    """
    
    def __init__(self, input_dim: int = 64):
        """
        Initialize value head.
        
        Args:
            input_dim: Dimension of input features
        """
        super(ValueHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Evaluation tensor of shape [batch, 1] in range [-1, 1]
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class PolicyHead(nn.Module):
    """
    Policy head for move prediction.
    
    Outputs logits over all possible moves (1968 moves).
    """
    
    def __init__(self, input_dim: int = 64, num_moves: int = 1968):
        """
        Initialize policy head.
        
        Args:
            input_dim: Dimension of input features
            num_moves: Number of possible moves (default 1968 = 64×64 - 8)
        """
        super(PolicyHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_moves)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, input_dim]
            
        Returns:
            Logits tensor of shape [batch, num_moves]
        """
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

