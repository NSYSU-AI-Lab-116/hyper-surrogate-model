"""
Command Line Interface for Hyper Surrogate Model
"""
import sys
import subprocess
from pathlib import Path
from hypersurrogatemodel import Logger
logger = Logger("CLI")

exe_path = sys.executable

def train_acc():
    """Start new training"""
    script_path = Path(__file__).parent / "entries_acc" / "training.py"
    subprocess.run([exe_path, str(script_path)])

def train_rank():
    """Start new training"""
    script_path = Path(__file__).parent / "entries_rank" / "training.py"
    subprocess.run([exe_path, str(script_path)])

def evaluate():
    """Evaluate trained model"""
    script_path = Path(__file__).parent / "evaluate" / "evaluate.py"
    subprocess.run([exe_path, str(script_path)])

def data_process():
    """Process training data"""
    script_path = Path(__file__).parent/ "entries_acc" / "data_reconstruct.py"
    subprocess.run([exe_path, str(script_path)])

def train_test():
    """Test training script"""
    subprocess.run(["torchrun","--nproc_per_node=2","/home/alvin/hyper-surrogate-model/entries_acc/training.py"])