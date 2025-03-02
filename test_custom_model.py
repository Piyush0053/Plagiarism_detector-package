import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlagiarismDetector(torch.nn.Module):
    """Transformer-based model for plagiarism detection"""
    
    def __init__(self, model_name='microsoft/deberta-v3-small'):
        """
        Initialize model
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        super(PlagiarismDetector, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        
        # Get the hidden size from the transformer model
        hidden_size = self.transformer.config.hidden_size
        
        # Classification layers
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, 1)
    
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

def load_model(model_dir):
    """
    Load the trained plagiarism detection model
    
    Args:
        model_dir (str): Directory containing the trained model
        
    Returns:
        tuple: Model and tokenizer
    """
    # Check if model exists
    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'final_model.pt')
        if not os.path.exists(model_path):
            logger.error(f"No model found in {model_dir}")
            sys.exit(1)
    
    # Load model arguments
    model_args_path = os.path.join(model_dir, 'training_args.json')
    if os.path.exists(model_args_path):
        import json
        with open(model_args_path, 'r') as f:
            args = json.load(f)
        model_name = args.get('model_name', 'microsoft/deberta-v3-small')
    else:
        logger.warning(f"No training args found at {model_args_path}, using default model name")
        model_name = 'microsoft/deberta-v3-small'
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Initialize model
    logger.info(f"Loading model from {model_path}")
    model = PlagiarismDetector(model_name=model_name)
    
    # Load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def detect_plagiarism(text1, text2, model, tokenizer, device, threshold=0.5):
    """
    Detect plagiarism between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        model: Trained model
        tokenizer: Tokenizer
        device: Device to run inference on
        threshold (float): Threshold for plagiarism detection
        
    Returns:
        dict: Results containing score and boolean flag
    """
    logger.info("Detecting plagiarism...")
    
    # Tokenize texts
    encoding = tokenizer(
        text1, text2,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Run inference
    with torch.no_grad():
        score = model(input_ids, attention_mask)
        score = score.squeeze().item()
    
    # Determine if plagiarism is detected
    is_plagiarism = score >= threshold
    
    # Results
    results = {
        'score': score,
        'is_plagiarism': is_plagiarism,
        'threshold': threshold
    }
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the custom plagiarism detection model")
    parser.add_argument("--model_dir", type=str, default="web_app/model", 
                        help="Directory containing the trained model")
    parser.add_argument("--text1", type=str, required=True,
                        help="First text to compare")
    parser.add_argument("--text2", type=str, required=True,
                        help="Second text to compare")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for plagiarism detection")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_dir)
    
    # Detect plagiarism
    results = detect_plagiarism(
        text1=args.text1,
        text2=args.text2,
        model=model,
        tokenizer=tokenizer,
        device=device,
        threshold=args.threshold
    )
    
    # Print results
    logger.info(f"Plagiarism score: {results['score']:.4f}")
    logger.info(f"Is plagiarism: {results['is_plagiarism']}")
    
    # More detailed output
    if results['is_plagiarism']:
        logger.info("PLAGIARISM DETECTED!")
        logger.info(f"Text similarity exceeds threshold of {results['threshold']}")
    else:
        logger.info("No plagiarism detected.")
        logger.info(f"Text similarity below threshold of {results['threshold']}")

if __name__ == "__main__":
    main() 