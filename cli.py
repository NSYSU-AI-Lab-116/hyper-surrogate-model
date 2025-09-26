"""
Command Line Interface for Hyper Surrogate Model
"""
import sys
import subprocess
from pathlib import Path
from hypersurrogatemodel import Logger
logger = Logger("CLI")


def train_acc():
    """Start new training"""
    script_path = Path(__file__).parent / "entries_acc" / "training.py"
    subprocess.run([sys.executable, str(script_path)])

def train_rank():
    """Start new training"""
    script_path = Path(__file__).parent / "entries_rank" / "training.py"
    subprocess.run([sys.executable, str(script_path)])

def evaluate():
    """Evaluate trained model"""
    script_path = Path(__file__).parent / "evaluate" / "evaluate.py"
    subprocess.run([sys.executable, str(script_path)])

def data_process():
    """Process training data"""
    script_path = Path(__file__).parent/ "entries_acc" / "data_reconstruct.py"
    subprocess.run([sys.executable, str(script_path)])