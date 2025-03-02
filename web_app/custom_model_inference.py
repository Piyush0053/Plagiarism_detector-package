import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import logging
from model_training import PlagiarismDetector
from nlp_utils import preprocess_text_for_nlp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomPlagiarismModel:
    """Class for loading and using the custom-trained plagiarism detection model"""
    
    def __init__(self, model_dir='model', device=None):
        """
        Initialize the custom plagiarism model
        
        Args:
            model_dir (str): Directory containing the model and tokenizer
            device (str): Device to use for inference (cpu or cuda)
        """
        self.model_dir = model_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing custom plagiarism model from {model_dir} on {self.device}")
    
    def load(self):
        """Load the model and tokenizer"""
        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            logger.error(f"Model directory {self.model_dir} does not exist")
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist")
        
        # Check if model weights exist
        model_path = os.path.join(self.model_dir, "plagiarism_detector.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model weights file {model_path} does not exist")
            raise FileNotFoundError(f"Model weights file {model_path} does not exist")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Initialize model
            self.model = PlagiarismDetector()
            
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text1, text2, threshold=0.5):
        """
        Predict if text2 is plagiarized from text1
        
        Args:
            text1 (str): Original text
            text2 (str): Text to check for plagiarism
            threshold (float): Threshold for classification
            
        Returns:
            dict: Dictionary with plagiarism score and result
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            if not self.load():
                return {
                    "error": "Model could not be loaded",
                    "score": 0.0,
                    "is_plagiarized": False
                }
        
        try:
            # Tokenize inputs
            encoding = self.tokenizer(
                text1,
                text2,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move inputs to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Disable gradient calculation
            with torch.no_grad():
                # Forward pass
                outputs = self.model(**encoding)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get plagiarism probability (class 1)
                plagiarism_prob = probs[0, 1].item()
                
                # Determine if plagiarized
                is_plagiarized = plagiarism_prob >= threshold
            
            return {
                "score": plagiarism_prob,
                "is_plagiarized": bool(is_plagiarized),
                "threshold": threshold
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "error": str(e),
                "score": 0.0,
                "is_plagiarized": False
            }

def compare_with_custom_model(text1, text2, threshold=0.5, model_dir='model'):
    """
    Compare two texts using the custom plagiarism detection model
    
    Args:
        text1 (str): Original text
        text2 (str): Text to check for plagiarism
        threshold (float): Threshold for classification
        model_dir (str): Directory containing the model and tokenizer
        
    Returns:
        dict: Dictionary with plagiarism score and result
    """
    model = CustomPlagiarismModel(model_dir=model_dir)
    return model.predict(text1, text2, threshold=threshold) 