import matplotlib
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np 
import torch.utils.data as Data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data  import *
from model import *


class Framework():
    def __init__(self, args):
        self.device       = torch.device(args.device)
        self.epoch        = args.epoch
        self.batch_size   = args.batch_size
        self.data         = DataGenerator(args)
        self.model        = FCDenseNet(args.blocks, args.growth_rate, args.in_channels, args.n_classes, args.original).to(self.device)
        self.optimizer    = torch.optim.RMSprop(self.model.parameters(), lr = args.lr,  weight_decay = 0.0001)
        self.train_loader = None
        self.test_feature = None
        self.test_label   = None 
        self.writer       = SummaryWriter('runs/' + str(args.exp_id))
        self.exp_id       = args.exp_id
    def show_model(self):
        print(self.model)
        print("Trainable Parameters: %d"%sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def pixel_acc(self, pred, target):
        target = torch.argmax(target, dim = 1)
        _, tags = torch.max(pred, dim = 1)
        corrects = (tags == target).float()
        acc = corrects.sum() / corrects.numel()
        return acc

    def iou(self, pred, target, n_classes = 32):
        intersection = torch.sum(torch.abs(pred * target), dim = [1,2,3])
        union = torch.sum(pred, dim = [1,2,3]) + torch.sum(target, dim = [1,2,3]) - intersection
        return torch.mean((intersection +0.0000001) / (union+0.0000001), axis = 0)
        
    def CrossEntropyLoss(self, y_pred, y_true):
        y_true = torch.argmax(y_true, dim = 1)
        criterion = nn.CrossEntropyLoss(reduce = None)
        return criterion(y_pred, y_true)
    

    def dice_loss(self, pred, target):
        intersection = torch.sum(torch.abs(pred * target), dim = [1,2,3])
        union = torch.sum(pred, dim = [1,2,3]) + torch.sum(target, dim = [1,2,3]) - intersection
        dice = torch.mean((2.*intersection +0.0000001) / (union+0.00000001), axis = 0)
        return 1.-dice 

    def focal_loss(self, pred, target):
        target = torch.argmax(target, dim = 1)
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        pred = pred.transpose(1,2)
        pred = pred.contiguous().view(-1, pred.size(2)).squeeze()
        # print(pred.shape)
        target = target.view(-1)
        criterion = nn.CrossEntropyLoss()
        
        # print(target.shape)
        loss = -criterion(pred, target)
        exp_loss = torch.exp(loss)

        loss = -((1-exp_loss) ** 2) * loss

        return loss.mean()
    def train(self):
        iteration = 0
        for Epoch in tqdm(range(self.epoch)):
            for steps, (x, y) in enumerate(self.train_loader):
                self.model.train()
                out = self.model(x)
                
                if Epoch < 5:
                    loss = self.dice_loss(out, y) + self.CrossEntropyLoss(out, y)
                else:
                    loss = self.dice_loss(out, y) + self.CrossEntropyLoss(out, y) + self.focal_loss(out, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.model.eval()
                    train_iou = self.iou(out, y)
                    train_acc = self.pixel_acc(out, y)

                    out = self.model(self.test_feature)
                    
                    if Epoch < 5:
                        test_loss = self.dice_loss(out, self.test_label) + self.CrossEntropyLoss(out, self.test_label)
                    else:
                        test_loss = self.dice_loss(out, self.test_label) + self.CrossEntropyLoss(out, self.test_label) + self.focal_loss(out, self.test_label)

                    test_iou = self.iou(out, self.test_label)
                    test_acc = self.pixel_acc(out, self.test_label)
                    # print("Epoch: %3d || Steps: %3d\nTraining IOU: %.4f || Training Acc: %.4f || Training Loss: %.6f\nTesting  IOU: %.4f || Testing  Acc: %.4f || Testing  Loss: %.6f\n" %\
                    #                 (Epoch, steps, train_iou, train_acc, loss.cpu().detach().numpy(), test_iou, test_acc, test_loss.cpu().detach().numpy()))
                    
                    self.writer.add_scalar('Training IOU', train_iou, iteration)
                    self.writer.add_scalar('Training Pixel Accuracy', train_acc, iteration)
                    self.writer.add_scalar('Testing  IOU', test_iou, iteration)
                    self.writer.add_scalar('Testing  Pixel Accuracy', test_acc, iteration)

                    if Epoch < 5:
                        self.writer.add_scalar('Training loss', loss.item(), iteration)
                        self.writer.add_scalar('Testing  loss', test_loss.item(), iteration)
                    else:
                        self.writer.add_scalar('Training loss2', loss.item(), iteration)
                        self.writer.add_scalar('Testing  loss2', test_loss.item(), iteration)

                    if(steps % 5 == 0):
                        self.data.draw(self.test_feature[[4,6,-1]].detach().cpu(), self.test_label[[4,6,-1]].detach().cpu(), out[[4,6,-1]].detach().cpu())

                        if not os.path.exists('./../result/' + str(self.exp_id)):
                            os.mkdir('./../result/' + str(self.exp_id))

                        plt.savefig('./../result/' + str(self.exp_id)+'/image_' + str(Epoch) + "_" +str(steps) +".png")
                iteration+=1
 


    
    def get_data(self):

        train_feature, train_label, test_feature, test_label = self.data.get_data()

        train_feature, train_label= torch.FloatTensor(train_feature).to(self.device), torch.LongTensor(train_label).to(self.device)
        test_feature, test_label= torch.FloatTensor(test_feature).to(self.device), torch.LongTensor(test_label).to(self.device)

        torch_dataset  = Data.TensorDataset(train_feature,train_label)
        self.train_loader   = Data.DataLoader(
            dataset    = torch_dataset,
            batch_size = self.batch_size,
            shuffle    = True,
        )
        self.test_feature = test_feature
        self.test_label   = test_label



    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path'           , type = str   , default = "./../data/")
    parser.add_argument('--img_w'          , type = int   , default = 128)
    parser.add_argument('--img_h'          , type = int   , default = 128)
    
    args = parser.parse_args()