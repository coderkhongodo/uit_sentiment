#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tạo trực quan hóa kết quả huấn luyện từ training logs
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import TrainingVisualizer
import glob

def parse_training_log(log_file):
    """
    Parse training log file để extract metrics
    
    Args:
        log_file (str): Path to log file
        
    Returns:
        dict: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rates': [],
        'steps': [],
        'epochs': []
    }
    
    step_data = []
    epoch_data = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract step-by-step training data
    step_pattern = r'Step (\d+(?:,\d+)*) \| Loss: ([\d.]+) \| Acc: ([\d.]+) \| LR: ([\d.e-]+)'
    step_matches = re.findall(step_pattern, content)
    
    for match in step_matches:
        step = int(match[0].replace(',', ''))
        loss = float(match[1])
        acc = float(match[2])
        lr = float(match[3])
        
        step_data.append({
            'step': step,
            'loss': loss,
            'acc': acc,
            'lr': lr
        })
    
    # Extract epoch results
    epoch_pattern = r'EPOCH (\d+) TRAINING RESULTS:.*?Average Loss: ([\d.]+).*?Average Accuracy: ([\d.]+).*?Average F1: ([\d.]+).*?Average Precision: ([\d.]+).*?Average Recall: ([\d.]+)'
    epoch_matches = re.findall(epoch_pattern, content, re.MULTILINE | re.DOTALL)
    
    # Extract dev evaluation results
    dev_pattern = r'DEV EVALUATION RESULTS:.*?Accuracy: ([\d.]+).*?F1 Score: ([\d.]+).*?Precision: ([\d.]+).*?Recall: ([\d.]+)'
    dev_matches = re.findall(dev_pattern, content, re.MULTILINE | re.DOTALL)
    
    # Process epoch data
    for i, match in enumerate(epoch_matches):
        epoch = int(match[0])
        train_loss = float(match[1])
        train_acc = float(match[2])
        train_f1 = float(match[3])
        train_precision = float(match[4])
        train_recall = float(match[5])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        
        # Add corresponding dev results if available
        if i < len(dev_matches):
            val_acc = float(dev_matches[i][0])
            val_f1 = float(dev_matches[i][1])
            val_precision = float(dev_matches[i][2])
            val_recall = float(dev_matches[i][3])
            
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_loss'].append(0.0)  # Validation loss not logged
        
        # Add learning rate from step data
        epoch_steps = [s for s in step_data if s['step'] <= epoch * 700]  # Approximate steps per epoch
        if epoch_steps:
            avg_lr = sum(s['lr'] for s in epoch_steps[-50:]) / min(50, len(epoch_steps))
            history['learning_rates'].append(avg_lr)
    
    return history, step_data

def create_step_by_step_visualization(step_data, save_dir='saved_results/plots'):
    """
    Tạo visualization theo từng step
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not step_data:
        print("⚠️ Không có dữ liệu step để visualize")
        return
    
    df = pd.DataFrame(step_data)
    
    # Create step-by-step plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📈 Step-by-Step Training Progress', fontsize=16, fontweight='bold')
    
    # Loss over steps
    axes[0, 0].plot(df['step'], df['loss'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('📉 Training Loss Over Steps')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy over steps
    axes[0, 1].plot(df['step'], df['acc'], 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('🎯 Training Accuracy Over Steps')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate over steps
    axes[1, 0].plot(df['step'], df['lr'], 'r-', alpha=0.7, linewidth=1)
    axes[1, 0].set_title('⚙️ Learning Rate Schedule')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1, 1].hist(df['loss'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('📊 Loss Distribution')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'step_by_step_training.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Step-by-step plots saved to: {save_dir}/step_by_step_training.png")

def create_training_summary_dashboard(history, step_data, save_dir='saved_results/plots'):
    """
    Tạo dashboard tổng quan kết quả training
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('🚀 PhoBERT Sentiment Analysis - Training Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Training curves (epochs)
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=3, marker='o')
    ax1.plot(epochs, history['train_acc'], 'g-', label='Train Accuracy', linewidth=3, marker='s')
    ax1.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=3, marker='^')
    ax1.set_title('📈 Training Progress by Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Metric Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score comparison
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['Train F1', 'Val F1']
    final_f1 = [history['train_f1'][-1], history['val_f1'][-1]]
    colors = ['#3498db', '#e74c3c']
    bars = ax2.bar(metrics, final_f1, color=colors, alpha=0.8)
    ax2.set_title('🏆 Final F1 Scores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0.8, 1.0)
    for bar, value in zip(bars, final_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Learning rate schedule
    ax3 = fig.add_subplot(gs[0, 3])
    if step_data:
        df = pd.DataFrame(step_data)
        ax3.plot(df['step'], df['lr'], 'purple', linewidth=2)
        ax3.set_title('⚙️ Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Step-by-step loss (smoothed)
    ax4 = fig.add_subplot(gs[1, :2])
    if step_data:
        df = pd.DataFrame(step_data)
        # Smooth the loss curve
        window_size = min(50, len(df))
        if window_size > 1:
            smoothed_loss = df['loss'].rolling(window=window_size, center=True).mean()
            ax4.plot(df['step'], df['loss'], 'lightblue', alpha=0.3, label='Raw Loss')
            ax4.plot(df['step'], smoothed_loss, 'darkblue', linewidth=2, label='Smoothed Loss')
        else:
            ax4.plot(df['step'], df['loss'], 'darkblue', linewidth=2, label='Loss')
        ax4.set_title('📉 Training Loss Over Steps', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Accuracy improvement
    ax5 = fig.add_subplot(gs[1, 2])
    if len(history['train_acc']) > 1:
        acc_improvement = [history['train_acc'][i] - history['train_acc'][i-1] for i in range(1, len(history['train_acc']))]
        ax5.bar(range(2, len(history['train_acc'])+1), acc_improvement, 
               color=['green' if x >= 0 else 'red' for x in acc_improvement], alpha=0.7)
        ax5.set_title('📊 Accuracy Improvement', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy Change')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.grid(True, alpha=0.3)
    
    # 6. Training stats summary
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    # Calculate stats
    total_epochs = len(history['train_loss'])
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
    
    stats_text = f"""
📊 TRAINING SUMMARY
================
🔢 Total Epochs: {total_epochs}
🎯 Best Train Acc: {best_train_acc:.3f}
🎯 Best Val Acc: {best_val_acc:.3f}
📉 Final Loss: {final_train_loss:.3f}
🏆 Final Train F1: {history['train_f1'][-1]:.3f}
🏆 Final Val F1: {history['val_f1'][-1]:.3f}
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7. Metrics comparison heatmap
    ax7 = fig.add_subplot(gs[2, :2])
    
    metrics_data = []
    for i, epoch in enumerate(epochs):
        metrics_data.append([
            history['train_acc'][i],
            history['train_f1'][i], 
            history['train_precision'][i],
            history['train_recall'][i]
        ])
    
    metrics_df = pd.DataFrame(metrics_data, 
                             columns=['Accuracy', 'F1', 'Precision', 'Recall'],
                             index=[f'Epoch {i}' for i in epochs])
    
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
               center=0.9, vmin=0.8, vmax=1.0, ax=ax7, cbar_kws={'label': 'Score'})
    ax7.set_title('🎯 Training Metrics Heatmap', fontsize=14, fontweight='bold')
    
    # 8. Loss distribution
    ax8 = fig.add_subplot(gs[2, 2:])
    if step_data:
        df = pd.DataFrame(step_data)
        ax8.hist(df['loss'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax8.axvline(df['loss'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["loss"].mean():.3f}')
        ax8.axvline(df['loss'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["loss"].median():.3f}')
        ax8.set_title('📊 Training Loss Distribution', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Loss Value')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🚀 Training dashboard saved to: {save_dir}/training_dashboard.png")

def main():
    """
    Main function để tạo tất cả visualization
    """
    print("🎨 Bắt đầu tạo trực quan hóa kết quả huấn luyện...")
    
    # Tìm file log mới nhất
    log_files = glob.glob('saved_results/training_*.log')
    if not log_files:
        print("❌ Không tìm thấy file training log!")
        return
    
    # Sử dụng file log mới nhất
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"📄 Đang parse file: {latest_log}")
    
    # Parse training log
    history, step_data = parse_training_log(latest_log)
    
    if not history['train_loss']:
        print("❌ Không tìm thấy dữ liệu training trong log file!")
        return
    
    print(f"✅ Đã parse được {len(history['train_loss'])} epochs và {len(step_data)} steps")
    
    # Tạo thư mục plots
    plots_dir = 'saved_results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Tạo các visualization
    print("🎨 Đang tạo training dashboard...")
    create_training_summary_dashboard(history, step_data, plots_dir)
    
    print("📊 Đang tạo step-by-step visualization...")
    create_step_by_step_visualization(step_data, plots_dir)
    
    # Sử dụng TrainingVisualizer có sẵn
    visualizer = TrainingVisualizer(plots_dir)
    
    if len(history['train_loss']) > 0:
        print("📈 Đang tạo training curves...")
        visualizer.plot_training_curves(history)
    
    # Lưu history data
    history_file = os.path.join(plots_dir, 'training_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Training history saved to: {history_file}")
    print(f"✅ Hoàn thành! Tất cả plots được lưu trong: {plots_dir}")
    print(f"📁 Các file đã tạo:")
    for file in os.listdir(plots_dir):
        if file.endswith('.png'):
            print(f"   📊 {file}")

if __name__ == "__main__":
    main() 