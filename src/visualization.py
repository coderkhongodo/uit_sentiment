# -*- coding: utf-8 -*-
"""
Visualization module cho training metrics và evaluation
- Training curves (loss, accuracy, F1)
- Confusion matrix heatmap
- Per-class performance charts
- Final evaluation plots
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Setup matplotlib cho tiếng Việt
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """
    Class để tạo các visualization cho training results
    """
    
    def __init__(self, save_dir='saved_results/plots'):
        """
        Khởi tạo visualizer
        
        Args:
            save_dir (str): Directory để lưu plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
    
    def plot_training_curves(self, history, save_path=None):
        """
        Vẽ training curves (loss, accuracy, F1)
        
        Args:
            history (dict): Training history
            save_path (str): Path để lưu plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_curves.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Training Progress Overview', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('📉 Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('🎯 Accuracy Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2)
        axes[1, 0].set_title('🏆 F1-Score Over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate plot
        axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 1].set_title('⚙️ Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved to: {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", save_path=None):
        """
        Vẽ confusion matrix heatmap
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            title (str): Plot title
            save_path (str): Path để lưu plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'📊 {title}', fontsize=16, fontweight='bold')
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Counts')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Oranges',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage'})
        ax2.set_title('Percentages')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {save_path}")
        
        return cm
    
    def plot_classification_report(self, y_true, y_pred, save_path=None):
        """
        Vẽ classification report heatmap
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            save_path (str): Path để lưu plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'classification_report.png')
        
        # Get classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        df = df.iloc[:3]  # Only class-specific metrics
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, vmin=0, vmax=1, ax=ax,
                   cbar_kws={'label': 'Score'})
        
        ax.set_title('🎯 Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎯 Classification report saved to: {save_path}")
    
    def plot_class_distribution(self, y_true, title="Class Distribution", save_path=None):
        """
        Vẽ class distribution
        
        Args:
            y_true (list): True labels
            title (str): Plot title
            save_path (str): Path để lưu plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'class_distribution.png')
        
        # Count classes
        unique, counts = np.unique(y_true, return_counts=True)
        percentages = counts / len(y_true) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'📊 {title}', fontsize=14, fontweight='bold')
        
        # Bar plot
        bars = ax1.bar([self.class_names[i] for i in unique], counts, 
                      color=[self.colors[i] for i in unique], alpha=0.8)
        ax1.set_title('Sample Counts')
        ax1.set_ylabel('Number of Samples')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(percentages, labels=[f'{name}\n({pct:.1f}%)' 
                                   for name, pct in zip([self.class_names[i] for i in unique], percentages)],
               colors=[self.colors[i] for i in unique], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Percentage Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Class distribution saved to: {save_path}")
    
    def create_training_summary(self, results_path, save_path=None):
        """
        Tạo summary visualization từ training results
        
        Args:
            results_path (str): Path to training_results.json
            save_path (str): Path để lưu summary plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_summary.png')
        
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        history = results['history']
        config = results['config']
        
        # Create summary plot
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('🎉 Training Summary Dashboard', fontsize=18, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss subplot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('📉 Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy subplot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('🎯 Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score subplot
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
        ax3.plot(epochs, history['val_f1'], 'r-', label='Validation', linewidth=2)
        ax3.set_title('🏆 F1-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training info
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Training statistics
        training_time = results['total_training_time']
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        info_text = f"""
📋 TRAINING CONFIGURATION:
   • Model: PhoBERT-base
   • Epochs: {config['epochs']} (completed: {results['final_epoch']})
   • Batch Size: {config['batch_size']}
   • Learning Rate: {config['learning_rate']}
   • Total Samples: {results['total_samples_seen']:,}

⏱️ TRAINING STATISTICS:
   • Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}
   • Best Validation F1: {results['best_val_f1']:.4f}
   • Final Train Loss: {history['train_loss'][-1]:.4f}
   • Final Val Loss: {history['val_loss'][-1]:.4f}
   • Final Train Accuracy: {history['train_acc'][-1]:.4f}
   • Final Val Accuracy: {history['val_acc'][-1]:.4f}
        """
        
        ax4.text(0.05, 0.5, info_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Epoch times
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(epochs, history['epoch_times'], color='skyblue', alpha=0.7)
        ax5.set_title('⏱️ Epoch Duration')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Time (seconds)')
        ax5.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax6.set_title('⚙️ Learning Rate')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('LR')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Best metrics
        ax7 = fig.add_subplot(gs[2, 2])
        metrics = ['Train F1', 'Val F1', 'Train Acc', 'Val Acc']
        values = [
            max(history['train_f1']),
            max(history['val_f1']),
            max(history['train_acc']),
            max(history['val_acc'])
        ]
        bars = ax7.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax7.set_title('🎯 Best Metrics')
        ax7.set_ylabel('Score')
        ax7.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎉 Training summary saved to: {save_path}")


def visualize_training_results(results_dir='saved_results'):
    """
    Tạo tất cả visualizations từ training results
    
    Args:
        results_dir (str): Directory chứa training results
    """
    print("🎨 Creating training visualizations...")
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir=os.path.join(results_dir, 'plots'))
    
    # Load training results
    results_path = os.path.join(results_dir, 'training_results.json')
    
    if not os.path.exists(results_path):
        print(f"❌ Training results not found: {results_path}")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create visualizations
    history = results['history']
    
    # 1. Training curves
    visualizer.plot_training_curves(history)
    
    # 2. Training summary dashboard
    visualizer.create_training_summary(results_path)
    
    print("✅ All visualizations created successfully!")
    print(f"📁 Saved to: {visualizer.save_dir}")


if __name__ == "__main__":
    # Example usage
    visualize_training_results() 