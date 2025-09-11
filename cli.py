"""
Command Line Interface for Hyper Surrogate Model
"""
import sys
import subprocess
from pathlib import Path

def train_new():
    """Start new training"""
    script_path = Path(__file__).parent / "entries" / "init_training.py"
    subprocess.run([sys.executable, str(script_path)])

def train_continue():
    """Continue training from saved model"""
    script_path = Path(__file__).parent / "entries" / "continue_training.py"
    subprocess.run([sys.executable, str(script_path)])

def evaluate():
    """Evaluate trained model"""
    script_path = Path(__file__).parent / "example" / "evaluate.py"
    subprocess.run([sys.executable, str(script_path)])

def data_process():
    """Process training data"""
    script_path = Path(__file__).parent/ "entries" / "data_reconstruct.py"
    subprocess.run([sys.executable, str(script_path)])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyper Surrogate Model CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('train_new', help='Start new training')
    subparsers.add_parser('train', help='Continue training')
    subparsers.add_parser('eval', help='Evaluate model')
    subparsers.add_parser('data', help='Process data')
    
    args = parser.parse_args()
    
    if args.command == 'train_new':
        train_new()
    elif args.command == 'train':
        train_continue()
    elif args.command == 'eval':
        evaluate()
    elif args.command == 'data':
        data_process()
    else:
        parser.print_help()