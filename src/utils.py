# Hàm tiện ích (đọc file, xử lý text, log...) 

# -*- coding: utf-8 -*-
"""
Utility functions cho training và evaluation
- Logging setup
- Early stopping
- Metrics computation
- Checkpoint handling
"""

import os
import logging
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def setup_logging(log_dir):
    """
    Setup logging với file và console output
    
    Args:
        log_dir (str): Directory để lưu log files
        
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('PhoBERT_Training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Logging setup completed. Log file: {log_file}")
    
    return logger


class EarlyStopping:
    """
    Early stopping để ngăn overfitting
    """
    
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): Số epochs chờ trước khi stop
            min_delta (float): Minimum improvement threshold
            restore_best_weights (bool): Có restore best weights không
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model=None):
        """
        Check nếu nên early stop
        
        Args:
            score (float): Current validation score (higher is better)
            model (torch.nn.Module): Model để save best weights
            
        Returns:
            bool: True nếu nên stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


def compute_metrics(predictions, labels, average='weighted'):
    """
    Tính toán các metrics cho classification
    
    Args:
        predictions (list): Predicted labels
        labels (list): True labels
        average (str): Averaging method cho F1 score
        
    Returns:
        dict: Dictionary chứa các metrics
    """
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'per_class': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        }
    }


def save_training_log(log_data, save_path):
    """
    Lưu training log thành JSON
    
    Args:
        log_data (dict): Dữ liệu log
        save_path (str): Đường dẫn file để lưu
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy arrays to lists nếu có
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    log_data = convert_numpy(log_data)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def format_time(seconds):
    """
    Format time thành human-readable string
    
    Args:
        seconds (float): Số giây
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def print_model_summary(model):
    """
    In summary của model
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("🤖 MODEL SUMMARY:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"   Model size: {size_mb:.1f} MB")


class AverageMeter:
    """
    Class để tính average của các metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    """
    Set random seed cho reproducibility
    
    Args:
        seed (int): Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get best available device
    
    Returns:
        torch.device: Best device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️ Using GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("🖥️ Using CPU")
    
    return device


def load_config(config_path):
    """
    Load configuration từ JSON file
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def save_config(config, save_path):
    """
    Save configuration to JSON file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False) 