# -*- coding: utf-8 -*-
"""
Module xử lý dữ liệu và tạo DataLoader cho việc training PhoBERT
- Load dữ liệu từ file .pkl đã tiền xử lý
- Xử lý class imbalance
- Tạo DataLoader hiệu quả
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter


class SentimentDataset(Dataset):
    """
    Enhanced dataset class for sentiment analysis
    Compatible with advanced preprocessing pipeline
    """
    
    def __init__(self, data_path=None, data_dict=None):
        """
        Initialize dataset from either file path or data dictionary
        
        Args:
            data_path (str): Path to .pkl file
            data_dict (dict): Data dictionary with required keys
        """
        if data_path:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        elif data_dict:
            self.data = data_dict
        else:
            raise ValueError("Either data_path or data_dict must be provided")
        
        # Validate required keys
        required_keys = ['input_ids', 'attention_mask', 'sentiments']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing required key: {key}")
        
        # Convert to tensors if not already
        if not isinstance(self.data['input_ids'], torch.Tensor):
            self.data['input_ids'] = torch.tensor(self.data['input_ids'], dtype=torch.long)
        if not isinstance(self.data['attention_mask'], torch.Tensor):
            self.data['attention_mask'] = torch.tensor(self.data['attention_mask'], dtype=torch.long)
        if not isinstance(self.data['sentiments'], torch.Tensor):
            self.data['sentiments'] = torch.tensor(self.data['sentiments'], dtype=torch.long)
        
        print(f"Dataset initialized with {len(self.data['sentiments'])} samples")
        
    def __len__(self):
        return len(self.data['sentiments'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['sentiments'][idx]
        }
    
    def get_class_distribution(self):
        """Get class distribution statistics"""
        sentiments = self.data['sentiments'].numpy()
        unique, counts = np.unique(sentiments, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print(f"Class distribution:")
        labels = ['Negative', 'Neutral', 'Positive']
        for i, label in enumerate(labels):
            count = distribution.get(i, 0)
            percentage = (count / len(sentiments)) * 100
            print(f"  {label} ({i}): {count} samples ({percentage:.1f}%)")
        
        return distribution


def create_data_loader(dataset, batch_size=16, shuffle=False, sampler_type='sequential'):
    """
    Create DataLoader with specified configuration
    Following reference notebook approach with SequentialSampler
    
    Args:
        dataset: SentimentDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data (ignored if sampler_type is specified)
        sampler_type: Type of sampler ('sequential', 'random')
    """
    if sampler_type == 'sequential':
        sampler = SequentialSampler(dataset)
    elif sampler_type == 'random':
        sampler = RandomSampler(dataset)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def load_all_dataloaders(batch_size=16, data_dir="data/processed"):
    """
    Load all dataloaders for train, dev, and test
    Following reference notebook configuration
    
    Args:
        batch_size: Batch size (default 16 as per reference)
        data_dir: Directory containing processed data files
    
    Returns:
        dict: Dictionary containing train, dev, test dataloaders
    """
    dataloaders = {}
    datasets = {}
    
    # Load datasets
    for split in ['train', 'dev', 'test']:
        try:
            file_path = f"{data_dir}/{split}.pkl"
            dataset = SentimentDataset(data_path=file_path)
            datasets[split] = dataset
            
            # Create appropriate dataloader
            if split == 'train':
                # Use SequentialSampler for training as per reference notebook
                dataloader = create_data_loader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler_type='sequential'
                )
            else:
                # Use SequentialSampler for validation/test
                dataloader = create_data_loader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler_type='sequential'
                )
            
            dataloaders[split] = dataloader
            
            print(f"\n{split.upper()} DataLoader:")
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Batch size: {batch_size}")
            print(f"  Number of batches: {len(dataloader)}")
            
            # Print class distribution
            dataset.get_class_distribution()
            
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping {split} dataset.")
        except Exception as e:
            print(f"Error loading {split} dataset: {e}")
    
    return dataloaders, datasets


def create_tensor_dataset(input_ids, attention_masks, labels):
    """
    Create TensorDataset for direct tensor inputs
    Compatible with reference notebook approach
    """
    # Convert to tensors if needed
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if not isinstance(attention_masks, torch.Tensor):
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    
    return TensorDataset(input_ids, attention_masks, labels)


def compute_class_weights(labels, device=None):
    """
    Compute class weights for imbalanced dataset
    Returns class weights as tensor
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Compute class weights
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    # Convert to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    if device:
        class_weights_tensor = class_weights_tensor.to(device)
    
    print("Class weights computed:")
    labels_names = ['Negative', 'Neutral', 'Positive']
    for i, weight in enumerate(class_weights):
        print(f"  {labels_names[i]}: {weight:.4f}")
    
    return class_weights_tensor


def test_dataloader():
    """Test dataloader functionality"""
    print("=== Testing DataLoader ===")
    
    try:
        # Load all dataloaders
        dataloaders, datasets = load_all_dataloaders(batch_size=16)
        
        # Test each dataloader
        for split, dataloader in dataloaders.items():
            print(f"\nTesting {split} dataloader:")
            
            # Get first batch
            batch = next(iter(dataloader))
            
            print(f"  Batch keys: {batch.keys()}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Labels sample: {batch['labels'][:5]}")
            
            # Verify data types
            assert batch['input_ids'].dtype == torch.long
            assert batch['attention_mask'].dtype == torch.long
            assert batch['labels'].dtype == torch.long
            
            print(f"  ✅ Data types correct")
            
        # Test class weights computation
        if 'train' in datasets:
            train_labels = datasets['train'].data['sentiments']
            class_weights = compute_class_weights(train_labels)
            print(f"\nClass weights shape: {class_weights.shape}")
        
        print("\n✅ All DataLoader tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False


if __name__ == "__main__":
    test_dataloader() 