import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model
    
    def load_model(self, path, epoch):
        try:
            state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch), 
                                  map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            raise e

    def save_model(self, path, epoch, acc, loss):
        try:
            torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, 
                      path+'{}_epoch_{}.pth'.format(self.name, epoch))
        except Exception as e:
            print(f"Warning: Could not save model to {path}: {e}")
            raise e

    def compute_loss(self, output, target, meta_target, meta_structure):
        pass

    def train_(self, image, target, meta_target, meta_structure, embedding, indicator):
        self.optimizer.zero_grad()
        output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().item()  # Use .item() for Python 3 compatibility
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().item()  # Use .item() for Python 3 compatibility
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().item()  # Use .item() for Python 3 compatibility
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy