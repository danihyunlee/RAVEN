#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all the imports used in main.py"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, utils
        print("✓ PyTorch imports successful")
        
        # Test utility imports
        from utility.dataset_utility import dataset, ToTensor
        print("✓ Utility imports successful")
        
        # Test model imports
        from cnn_mlp import CNN_MLP
        from resnet18 import Resnet18_MLP
        from basic_model import BasicModel
        from fc_tree_net import FCTreeNet
        print("✓ Model imports successful")
        
        # Test PIL import
        from PIL import Image
        print("✓ PIL import successful")
        
        print("\n🎉 All imports successful! The code should work now.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 