import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json
import time
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class PlagiarismDataset(Dataset):
    """Dataset for plagiarism detection"""
    
    def __init__(self, texts1, texts2, labels, tokenizer, max_length=512):
        """
        Initialize dataset
        
        Args:
            texts1 (list): List of original texts
            texts2 (list): List of possibly plagiarized texts
            labels (list): List of labels (1 for plagiarism, 0 for non-plagiarism)
            tokenizer: Tokenizer to use
            max_length (int): Maximum sequence length
        """
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = self.labels[idx]
        
        # Tokenize both texts
        encoding = self.tokenizer(
            text1, text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class PlagiarismDetector(nn.Module):
    """Transformer-based model for plagiarism detection"""
    
    def __init__(self, model_name='microsoft/deberta-v3-small'):
        """
        Initialize model
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        super(PlagiarismDetector, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Get the hidden size from the transformer model
        hidden_size = self.transformer.config.hidden_size
        
        # Classification layers
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Predict similarity score (0 to 1)
        logits = self.classifier(pooled_output)
        
        return torch.sigmoid(logits)

def load_data(data_path):
    """
    Load data from CSV file containing pairs of texts and plagiarism labels
    
    Expected CSV format:
    text1, text2, label
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_columns = ['text1', 'text2', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV file")
    
    # Print data statistics
    logger.info(f"Data statistics:")
    logger.info(f"  Number of pairs: {len(df)}")
    logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    logger.info(f"Training set: {train_df.shape[0]} samples")
    logger.info(f"Validation set: {val_df.shape[0]} samples")
    
    return train_df, val_df

def train_model(
    data_path, 
    model_name='microsoft/deberta-v3-small',
    output_dir='model',
    batch_size=8,
    epochs=3,
    learning_rate=2e-5,
    max_length=512,
    save_steps=1000
):
    """
    Train a plagiarism detection model using the provided data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training arguments
    training_args = {
        'model_name': model_name,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'data_path': data_path,
        'device': str(device),
        'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_args, f, indent=2)
    
    # Load and prepare data
    train_df, val_df = load_data(data_path)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = PlagiarismDataset(
        train_df['text1'].tolist(),
        train_df['text2'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )
    
    val_dataset = PlagiarismDataset(
        val_df['text1'].tolist(),
        val_df['text2'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=max_length
    )
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = PlagiarismDetector(model_name=model_name)
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    step = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            step += 1
            
            # Save checkpoint periodically
            if step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f'checkpoint-{step}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    return model, tokenizer

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            # Update metrics
            val_loss += loss.item()
            
            # Convert outputs to predictions
            preds = (outputs.squeeze() > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return avg_val_loss, accuracy, precision, recall, f1

def plot_training_history(history, output_dir):
    """
    Plot training history
    
    Args:
        history (dict): Training history
        output_dir (str): Directory to save plots
    """
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot confusion matrix
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        output_dir (str): Directory to save plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Plagiarized', 'Plagiarized'],
                yticklabels=['Not Plagiarized', 'Plagiarized'])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer to the specified directory
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train a plagiarism detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing text pairs and labels')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small', help='Name of the pre-trained model to use')
    parser.add_argument('--output_dir', type=str, default='model', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save model every N steps')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps
    )

if __name__ == "__main__":
    main() 