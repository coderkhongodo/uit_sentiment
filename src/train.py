"""
Advanced Training Pipeline for PhoBERT Sentiment Analysis
Based on reference notebook: uit-vsfc-phobert-base.ipynb (93.9% accuracy)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from data_loader import load_all_dataloaders, compute_class_weights
from model import PhoBERTSentimentClassifier
from utils import setup_logging, EarlyStopping, format_time

class AdvancedTrainer:
    """
    Advanced trainer based on reference notebook implementation
    Achieved 93.9% test accuracy in reference
    """
    
    def __init__(self, config=None):
        """Initialize trainer with optimized configuration"""
        # Default configuration based on reference notebook
        self.config = {
            'epochs': 3,                    # Reference used 20 epochs
            'batch_size': 16,               # Reference used 16
            'learning_rate': 2e-5,          # Reference used 2e-5  
            'weight_decay': 0.01,           # Reference used 0.01
            'warmup_steps': 0,              # Reference used 0 warmup steps
            'max_grad_norm': 1.0,           # Gradient clipping
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_steps': 500,
            'eval_steps': 100,
            'logging_steps': 50,
            'save_dir': 'saved_results',
            'model_name': 'vinai/phobert-base',
            'use_class_weights': True,
            'early_stopping_patience': 5    # More patience for longer training
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.device = torch.device(self.config['device'])
        self.logger = setup_logging(self.config['save_dir'])
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = {}
        self.datasets = {}
        
        # Training state
        self.global_step = 0
        self.best_eval_accuracy = 0.0
        self.training_stats = []
        
        print("=== Advanced PhoBERT Trainer Initialized ===")
        print(f"Configuration based on reference notebook (93.9% accuracy)")
        self._print_config()
    
    def _print_config(self):
        """Print training configuration"""
        print("\nTraining Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print()
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        print("Setting up model and training components...")
        
        # Initialize model
        self.model = PhoBERTSentimentClassifier(
            num_labels=3,
            model_name=self.config['model_name']
        )
        
        # Load model
        success = self.model.load_model()
        if not success:
            raise RuntimeError("Failed to load model")
        
        # Move to device
        self.model.model.to(self.device)
        
        # Setup optimizer (following reference notebook)
        param_optimizer = list(self.model.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.config['learning_rate']
        )
        
        print(f"✅ Model and optimizer setup complete")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.model.parameters() if p.requires_grad):,}")
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        self.dataloaders, self.datasets = load_all_dataloaders(
            batch_size=self.config['batch_size']
        )
        
        if 'train' not in self.dataloaders:
            raise RuntimeError("Training data not found")
        
        # Setup scheduler based on training data
        total_steps = len(self.dataloaders['train']) * self.config['epochs']
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Setup loss function with class weights if enabled
        if self.config['use_class_weights']:
            train_labels = self.datasets['train'].data['sentiments']
            class_weights = compute_class_weights(train_labels, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ Data setup complete")
        print(f"   Total training steps: {total_steps:,}")
        print(f"   Steps per epoch: {len(self.dataloaders['train']):,}")
    
    def train_epoch(self, epoch):
        """Train one epoch following reference notebook approach"""
        self.model.model.train()
        
        total_loss = 0
        train_accuracy = 0
        train_f1 = 0
        train_precision = 0
        train_recall = 0
        nb_train_steps = 0
        
        # Progress bar for epoch
        epoch_iterator = tqdm(
            self.dataloaders['train'], 
            desc=f"Epoch {epoch+1}/{self.config['epochs']}",
            leave=False
        )
        
        start_time = time.time()
        
        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.model.model.zero_grad()
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Calculate metrics
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            tmp_accuracy, tmp_f1, tmp_precision, tmp_recall = self.flat_accuracy(logits, label_ids)
            train_accuracy += tmp_accuracy
            train_f1 += tmp_f1
            train_precision += tmp_precision
            train_recall += tmp_recall
            nb_train_steps += 1
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            self.scheduler.step()
            
            # Update global step
            self.global_step += 1
            
            # Update progress bar
            epoch_iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{tmp_accuracy:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config['logging_steps'] == 0:
                self.logger.info(
                    f"Step {self.global_step:,} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {tmp_accuracy:.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )
        
        # Calculate epoch metrics
        avg_train_loss = total_loss / nb_train_steps
        avg_train_accuracy = train_accuracy / nb_train_steps
        avg_train_f1 = train_f1 / nb_train_steps
        avg_train_precision = train_precision / nb_train_steps
        avg_train_recall = train_recall / nb_train_steps
        
        epoch_time = time.time() - start_time
        
        # Log epoch results
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"EPOCH {epoch+1} TRAINING RESULTS:")
        self.logger.info(f"  Average Loss: {avg_train_loss:.4f}")
        self.logger.info(f"  Average Accuracy: {avg_train_accuracy:.4f}")
        self.logger.info(f"  Average F1: {avg_train_f1:.4f}")
        self.logger.info(f"  Average Precision: {avg_train_precision:.4f}")
        self.logger.info(f"  Average Recall: {avg_train_recall:.4f}")
        self.logger.info(f"  Epoch Time: {format_time(epoch_time)}")
        self.logger.info(f"{'='*50}")
        
        return {
            'loss': avg_train_loss,
            'accuracy': avg_train_accuracy,
            'f1': avg_train_f1,
            'precision': avg_train_precision,
            'recall': avg_train_recall,
            'time': epoch_time
        }
    
    def evaluate(self, eval_dataloader, split_name="dev"):
        """Evaluate model following reference notebook approach"""
        self.model.model.eval()
        
        eval_accuracy = 0
        eval_f1 = 0
        eval_precision = 0
        eval_recall = 0
        nb_eval_steps = 0
        
        all_preds = []
        all_labels = []
        
        print(f"\nEvaluating on {split_name} set...")
        
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            # Store for detailed metrics
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds)
            all_labels.extend(label_ids)
            
            tmp_accuracy, tmp_f1, tmp_precision, tmp_recall = self.flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_accuracy
            eval_f1 += tmp_f1
            eval_precision += tmp_precision
            eval_recall += tmp_recall
            nb_eval_steps += 1
        
        # Calculate final metrics
        avg_accuracy = eval_accuracy / nb_eval_steps
        avg_f1 = eval_f1 / nb_eval_steps
        avg_precision = eval_precision / nb_eval_steps
        avg_recall = eval_recall / nb_eval_steps
        
        # Detailed classification report
        try:
            report = classification_report(
                all_labels, all_preds,
                target_names=['Negative', 'Neutral', 'Positive'],
                output_dict=True,
                zero_division=0
            )
            
            # Log detailed results
            self.logger.info(f"\n{split_name.upper()} EVALUATION RESULTS:")
            self.logger.info(f"  Accuracy: {avg_accuracy:.4f}")
            self.logger.info(f"  F1 Score: {avg_f1:.4f}")
            self.logger.info(f"  Precision: {avg_precision:.4f}")
            self.logger.info(f"  Recall: {avg_recall:.4f}")
            self.logger.info(f"\nPer-class metrics:")
            
            for i, class_name in enumerate(['Negative', 'Neutral', 'Positive']):
                if str(i) in report:
                    self.logger.info(
                        f"  {class_name}: "
                        f"P={report[str(i)]['precision']:.4f}, "
                        f"R={report[str(i)]['recall']:.4f}, "
                        f"F1={report[str(i)]['f1-score']:.4f}"
                    )
                else:
                    self.logger.info(f"  {class_name}: NOT PREDICTED")
                    
        except Exception as e:
            self.logger.error(f"Error in classification report: {e}")
        
        return {
            'accuracy': avg_accuracy,
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'all_preds': all_preds,
            'all_labels': all_labels
        }
    
    def flat_accuracy(self, preds, labels):
        """Calculate accuracy and metrics like reference notebook"""
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        accuracy = accuracy_score(pred_flat, labels_flat)
        f1 = f1_score(pred_flat, labels_flat, average='weighted', zero_division=0)
        precision = precision_score(pred_flat, labels_flat, average='weighted', zero_division=0)
        recall = recall_score(pred_flat, labels_flat, average='weighted', zero_division=0)
        
        return accuracy, f1, precision, recall
    
    def save_checkpoint(self, epoch, eval_results):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['save_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}')
        self.model.save_model(checkpoint_path)
        
        # Save training state
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_eval_accuracy': self.best_eval_accuracy,
            'eval_results': eval_results,
            'config': self.config
        }
        
        state_path = os.path.join(checkpoint_path, 'training_state.pt')
        torch.save(state, state_path)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def train(self):
        """Main training loop following reference notebook"""
        print("\n=== Starting Training ===")
        print(f"Based on reference notebook configuration (achieved 93.9% accuracy)")
        
        # Setup training components
        self.setup_model()
        self.setup_data()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training loop
        total_start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*60}")
            
            # Train epoch
            train_results = self.train_epoch(epoch)
            
            # Evaluate on dev set
            if 'dev' in self.dataloaders:
                eval_results = self.evaluate(self.dataloaders['dev'], 'dev')
                
                # Check for best model
                if eval_results['accuracy'] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_results['accuracy']
                    self.save_checkpoint(epoch, eval_results)
                    print(f"🎉 New best model! Accuracy: {eval_results['accuracy']:.4f}")
                
                # Early stopping check (note: EarlyStopping expects higher score = better)
                should_stop = early_stopping(eval_results['accuracy'], self.model.model)
                
                if should_stop:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Store training stats
            self.training_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_results['loss'],
                'train_accuracy': train_results['accuracy'],
                'train_f1': train_results['f1'],
                'eval_accuracy': eval_results.get('accuracy', 0),
                'eval_f1': eval_results.get('f1', 0),
                'time': train_results['time']
            })
        
        # Final evaluation
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total training time: {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_eval_accuracy:.4f}")
        
        # Final test evaluation
        if 'test' in self.dataloaders:
            print(f"\n🧪 Final Test Evaluation:")
            test_results = self.evaluate(self.dataloaders['test'], 'test')
            
            print(f"\n🎯 FINAL TEST RESULTS:")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            print(f"   Test F1 Score: {test_results['f1']:.4f}")
            print(f"   Test Precision: {test_results['precision']:.4f}")
            print(f"   Test Recall: {test_results['recall']:.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.config['save_dir'], 'final_model')
        self.model.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        return self.training_stats

def main():
    """Main training function"""
    print("=== PhoBERT Advanced Training Pipeline ===")
    print("Based on reference notebook: uit-vsfc-phobert-base.ipynb")
    print("Reference achieved: 93.9% test accuracy")
    
    # Configuration optimized from reference notebook
    config = {
        'epochs': 3,            # Reduced for testing
        'batch_size': 16,       # Reference used 16
        'learning_rate': 2e-5,  # Reference used 2e-5
        'weight_decay': 0.01,   # Reference used 0.01
        'warmup_steps': 0,      # Reference used 0
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 5,
        'save_dir': 'saved_results',
        'use_class_weights': True
    }
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    
    # Start training
    try:
        training_stats = trainer.train()
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 