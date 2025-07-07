import os
import sys
import numpy as np
import argparse

# Add the current directory to Python path to find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utility.dataset_utility import dataset, ToTensor
from cnn_mlp import CNN_MLP, CNN_MLP_MIN
from resnet18 import Resnet18_MLP, Resnet18_MLP_MIN

parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='Resnet18_MLP')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default='./data/RAVEN-10000/')
parser.add_argument('--save', type=str, default='./experiments/checkpoint/')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_alpha', type=float, default=0.0)
parser.add_argument('--meta_beta', type=float, default=0.0)
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='RAVEN', help='WandB project name')
parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')


args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize wandb if enabled
if args.wandb:
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'model': args.model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'beta1': args.beta1,
                'beta2': args.beta2,
                'epsilon': args.epsilon,
                'meta_alpha': args.meta_alpha,
                'meta_beta': args.meta_beta,
                'img_size': args.img_size,
                'seed': args.seed,
            }
        )
        print(f"✅ WandB logging enabled for project: {args.wandb_project}")
    except ImportError:
        print("❌ WandB not installed. Install with: pip install wandb")
        args.wandb = False
    except Exception as e:
        print(f"❌ Failed to initialize WandB: {e}")
        args.wandb = False

# Device setup
args.cuda = torch.cuda.is_available()
if args.cuda:
    try:
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(f'cuda:{args.device}')
        print(f"Using CUDA device {args.device}")
    except Exception as e:
        print(f"Warning: Could not set CUDA device {args.device}: {e}")
        args.cuda = False
        device = torch.device('cpu')
        print("Falling back to CPU")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

if not os.path.exists(args.save):
    os.makedirs(args.save)

train = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]))
valid = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
test = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

# Use fewer workers if CUDA is not available or for better compatibility
num_workers = min(args.load_workers, 4) if not args.cuda else args.load_workers

trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

if args.model == "CNN_MLP":
    model = CNN_MLP(args)
elif args.model == "CNN_MLP_MIN":
    model = CNN_MLP_MIN(args)
elif args.model == "Resnet18_MLP":
    model = Resnet18_MLP(args)
elif args.model == "Resnet18_MLP_MIN":
    model = Resnet18_MLP_MIN(args)
    
if args.resume:
    try:
        model.load_model(args.save, 0)
        print('Loaded model')
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

# Move model to device
model = model.to(device)

def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(trainloader):
        counter += 1
        # Move data to device
        image = image.to(device)
        target = target.to(device)
        meta_target = meta_target.to(device)
        meta_structure = meta_structure.to(device)
        embedding = embedding.to(device)
        indicator = indicator.to(device)
        
        try:
            loss, acc = model.train_(image, target, meta_target, meta_structure, embedding, indicator)
            # Remove per-batch printing
            # print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
            loss_all += loss
            acc_all += acc
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
            
    if counter > 0:
        avg_loss = loss_all/float(counter)
        avg_acc = acc_all/float(counter)
        print('Train: Epoch:{}, Avg Loss:{:.6f}, Avg Acc:{:.4f}.'.format(epoch, avg_loss, avg_acc))
        
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                'train/loss': avg_loss,
                'train/accuracy': avg_acc,
                'epoch': epoch
            })
        
        return avg_loss, avg_acc
    return 0.0, 0.0

def validate(epoch):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(validloader):
            counter += 1
            # Move data to device
            image = image.to(device)
            target = target.to(device)
            meta_target = meta_target.to(device)
            meta_structure = meta_structure.to(device)
            embedding = embedding.to(device)
            indicator = indicator.to(device)
            
            try:
                loss, acc = model.validate_(image, target, meta_target, meta_structure, embedding, indicator)
                # Remove per-batch printing
                # print('Validate: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc)) 
                loss_all += loss
                acc_all += acc
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
                
    if counter > 0:
        avg_loss = loss_all/float(counter)
        avg_acc = acc_all/float(counter)
        print("Validate: Epoch:{}, Avg Loss: {:.6f}, Avg Acc: {:.4f}".format(epoch, avg_loss, avg_acc))
        
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/accuracy': avg_acc,
                'epoch': epoch
            })
        
        return avg_loss, avg_acc
    return 0.0, 0.0

def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    with torch.no_grad():
        for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(testloader):
            counter += 1
            # Move data to device
            image = image.to(device)
            target = target.to(device)
            meta_target = meta_target.to(device)
            meta_structure = meta_structure.to(device)
            embedding = embedding.to(device)
            indicator = indicator.to(device)
            
            try:
                acc = model.test_(image, target, meta_target, meta_structure, embedding, indicator)
                # Remove per-batch printing
                # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))  
                acc_all += acc
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue
                
    if counter > 0:
        avg_acc = acc_all/float(counter)
        print("Test: Epoch:{}, Avg Acc: {:.4f}".format(epoch, avg_acc))
        
        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                'test/accuracy': avg_acc,
                'epoch': epoch
            })
        
        return avg_acc
    return 0.0

def main():
    for epoch in range(0, args.epochs):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)
        test_acc = test(epoch)
        
        # Log combined metrics to wandb if enabled
        if args.wandb:
            wandb.log({
                'epoch_summary/train_loss': train_loss,
                'epoch_summary/train_accuracy': train_acc,
                'epoch_summary/val_loss': val_loss,
                'epoch_summary/val_accuracy': val_acc,
                'epoch_summary/test_accuracy': test_acc,
                'epoch': epoch
            })
        
        try:
            model.save_model(args.save, epoch, val_acc, val_loss)
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    # Final wandb logging
    if args.wandb:
        wandb.finish()
        print("✅ WandB logging completed")


if __name__ == '__main__':
    main()