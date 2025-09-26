"""
Command Line Interface for Hyper Surrogate Model
"""
import sys
import subprocess
from pathlib import Path
from hypersurrogatemodel import Logger
logger = Logger("CLI")


def train():
    """Start new training"""
    script_path = Path(__file__).parent / "entries_accuracy" / "training.py"
    subprocess.run([sys.executable, str(script_path)])

def evaluate():
    """Evaluate trained model"""
    script_path = Path(__file__).parent / "example" / "evaluate.py"
    subprocess.run([sys.executable, str(script_path)])

def data_process():
    """Process training data"""
    script_path = Path(__file__).parent/ "entries_accuracy" / "data_reconstruct.py"
    subprocess.run([sys.executable, str(script_path)])