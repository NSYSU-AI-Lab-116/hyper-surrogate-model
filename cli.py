"""
Command Line Interface for Hyper Surrogate Model
"""
import sys
import subprocess
from pathlib import Path
from hypersurrogatemodel.config import config
from hypersurrogatemodel import Logger
logger = Logger("CLI")

def train_new():
    """Start new training"""
    script_path = Path(__file__).parent / "entries" / "training_init.py"
    subprocess.run([sys.executable, str(script_path)])

def train_continue():
    """Continue training from saved model"""
    script_path = Path(__file__).parent / "entries" / "training_transfer.py"
    subprocess.run([sys.executable, str(script_path)])

def evaluate():
    """Evaluate trained model"""
    script_path = Path(__file__).parent / "example" / "evaluate.py"
    subprocess.run([sys.executable, str(script_path)])

def data_process():
    """Process training data"""
    script_path = Path(__file__).parent/ "entries" / "data_reconstruct.py"
    subprocess.run([sys.executable, str(script_path)])