import os
import sys
import subprocess
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_dataset():
    """
    Prepare the DAIGT dataset for training the plagiarism detection model
    """
    logger.info("Preparing dataset...")
    
    # Get the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Check if the dataset already exists
    dataset_file = os.path.join(parent_dir, "dataset", "plagiarism_dataset.csv")
    if os.path.exists(dataset_file):
        logger.info(f"Dataset already exists at {dataset_file}")
        return dataset_file
    
    # Run the script to prepare the dataset
    script_path = os.path.join(parent_dir, "prepare_daigt_dataset.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Dataset preparation script not found at {script_path}")
        sys.exit(1)
    
    logger.info("Running dataset preparation script...")
    try:
        subprocess.run([
            sys.executable, 
            script_path, 
            "--download",
            "--sample_size", "5000"  # Smaller sample size for faster training
        ], check=True)
        
        logger.info(f"Dataset prepared successfully at {dataset_file}")
        return dataset_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preparing dataset: {e}")
        sys.exit(1)

def train_model(data_path, output_dir="model", 
                model_name="microsoft/deberta-v3-base", 
                epochs=3, batch_size=4):
    """
    Train the plagiarism detection model
    """
    logger.info(f"Training model using {data_path}...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the training script
    train_script = os.path.join(current_dir, "model_training.py")
    
    if not os.path.exists(train_script):
        logger.error(f"Training script not found at {train_script}")
        sys.exit(1)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run the training script
        logger.info(f"Running training script: {train_script}")
        subprocess.run([
            sys.executable,
            train_script,
            "--data_path", data_path,
            "--output_dir", output_dir,
            "--model_name", model_name,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size)
        ], check=True)
        
        logger.info(f"Model trained successfully and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

def main():
    """
    Main function to run the training pipeline
    """
    parser = argparse.ArgumentParser(description="Train a plagiarism detection model on the DAIGT dataset")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small", 
                       help="Name of the pre-trained model to use")
    parser.add_argument("--output_dir", type=str, default="web_app/model", 
                       help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size for training")
    parser.add_argument("--skip_dataset_prep", action="store_true", 
                       help="Skip dataset preparation and use existing dataset")
    parser.add_argument("--existing_dataset", type=str, default=None,
                       help="Path to existing dataset (if skip_dataset_prep is True)")
    
    args = parser.parse_args()
    
    logger.info("Starting plagiarism model training pipeline...")
    
    # Prepare the dataset
    if args.skip_dataset_prep and args.existing_dataset:
        data_path = args.existing_dataset
        logger.info(f"Skipping dataset preparation, using existing dataset at {data_path}")
    else:
        data_path = prepare_dataset()
    
    # Train the model
    train_model(
        data_path=data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 