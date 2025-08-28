import argparse
import os
import torch
import random
import numpy as np

from utils.config import load_config, get_trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train porous material adsorption property prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # Not mandatory
        torch.backends.cudnn.benchmark = False # Not mandatory

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)

    # Create output directories
    os.makedirs(config.training.save_dir, exist_ok=True)

    # Get trainer and start training
    trainer = get_trainer(config)
    trainer.train()
    

if __name__ == '__main__':
    main() 