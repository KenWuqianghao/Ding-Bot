"""Ensemble evaluator - combines multiple NN models."""
import torch
import chess
import numpy as np
from typing import List
from .evaluation import NNEvaluator


class EnsembleEvaluator:
    """
    Ensemble of multiple neural network evaluators.
    Averages predictions from multiple models for better accuracy.
    """
    
    def __init__(self, models: List[torch.nn.Module], device: torch.device = None):
        """
        Initialize ensemble evaluator.
        
        Args:
            models: List of trained neural network models
            device: torch device
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Create evaluators for each model
        self.evaluators = [NNEvaluator(model, self.device) for model in models]
        print(f"Ensemble evaluator initialized with {len(self.evaluators)} models")
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using ensemble of models.
        
        Args:
            board: chess.Board object
            
        Returns:
            Average evaluation from all models (centipawns)
        """
        # Get evaluations from all models
        evaluations = []
        for evaluator in self.evaluators:
            try:
                eval_score = evaluator.evaluate(board)
                evaluations.append(eval_score)
            except Exception as e:
                print(f"Warning: Model evaluation failed: {e}")
                continue
        
        if not evaluations:
            # Fallback to material eval if all models fail
            return 0.0
        
        # Use median (more robust to outliers) or mean
        # Median is better if models disagree significantly
        if len(evaluations) >= 3:
            return float(np.median(evaluations))
        else:
            return float(np.mean(evaluations))
    
    def evaluate_with_std(self, board: chess.Board) -> tuple:
        """
        Evaluate position and return mean + standard deviation.
        
        Returns:
            (mean_eval, std_eval) tuple
        """
        evaluations = []
        for evaluator in self.evaluators:
            try:
                eval_score = evaluator.evaluate(board)
                evaluations.append(eval_score)
            except:
                continue
        
        if not evaluations:
            return 0.0, 0.0
        
        mean_eval = float(np.mean(evaluations))
        std_eval = float(np.std(evaluations))
        return mean_eval, std_eval

