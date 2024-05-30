import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  
from torch.utils.data import Dataset, DataLoader
from utils.BLSLoader import BLSRatingsDataset

import random
import numpy as np
from datetime import datetime
import logging
import sys


class BLS_Train():

    def __init__(self, train_dataset, model, device, batch_size = 1, learning_rate = 0.0001, weight_decay = 0.1, num_epochs = 100, seed = 30032023):

        self.train_dataset = train_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay * self.learning_rate
        self.num_epochs = num_epochs

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay)

        # set current time as output folder
        now = datetime.now()
        self.writer = SummaryWriter('runs/' + now.strftime("%Y%m%d-%H%M%S"))

        # set logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self):

        device = self.device

        best_loss = 1000000
        
        for epoch in range(self.num_epochs):
            loss_sum = 0
            for (i , inputs) in enumerate(self.train_loader):


                self.model.train()

                data = inputs['joint_all'].to(device)
                total_score = inputs['total_score'].to(device)
                hand_pos_score = inputs['Hand_Pos'].to(device)
                arm_pos_score = inputs['Arm_Pos'].to(device)
                sholder_pos_score = inputs['Shoulder_Pos'].to(device)
                depth_score = inputs['Depth'].to(device)
                rate_score = inputs['Rate'].to(device)
                release_score = inputs['Release'].to(device)


                self.optimizer.zero_grad()
                total, hand_pos, arm_pos, sholder_pos, depth, rate, release = self.model(data)
                loss = self.criterion(total, total_score) + \
                        self.criterion(hand_pos, hand_pos_score) + \
                        self.criterion(arm_pos, arm_pos_score) + \
                        self.criterion(sholder_pos, sholder_pos_score) + \
                        self.criterion(depth, depth_score) + \
                        self.criterion(rate, rate_score) + \
                        self.criterion(release, release_score)
                
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            # add loss to tensorboard
            self.writer.add_scalar('training loss', loss_sum, epoch)

            # add loss to log
            logging.info('Epoch: {}, Loss: {}'.format(epoch, loss_sum))
            # save model
            torch.save(self.model.state_dict(), 'model.pth')
                
            '''
            loss_val_sum = 0
            for (i , inputs) in enumerate(self.val_loader):

                self.model.eval()

                data = inputs['joint_all'].to(device)
                total_score = inputs['total_score'].to(device)
                hand_pos_score = inputs['Hand_Pos'].to(device)
                arm_pos_score = inputs['Arm_Pos'].to(device)
                sholder_pos_score = inputs['Shoulder_Pos'].to(device)
                depth_score = inputs['Depth'].to(device)
                rate_score = inputs['Rate'].to(device)
                release_score = inputs['Release'].to(device)

                with torch.no_grad():
                    total, hand_pos, arm_pos, sholder_pos, depth, rate, release = self.model(data)
                    loss = self.criterion(total, total_score) + \
                        self.criterion(hand_pos, hand_pos_score) + \
                        self.criterion(arm_pos, arm_pos_score) + \
                        self.criterion(sholder_pos, sholder_pos_score) + \
                        self.criterion(depth, depth_score) + \
                        self.criterion(rate, rate_score) + \
                        self.criterion(release, release_score)
                    loss_val_sum += loss.item()

            if loss_val_sum < best_loss:
                best_loss = loss_val_sum
                torch.save(self.model.state_dict(), 'best_model.pth')
                # add best model to log
                logging.info('Epoch: {}, Best Val Loss: {}'.format(epoch, best_loss))
            
            # add loss to tensorboard
            self.writer.add_scalar('validation loss', loss_val_sum, epoch)
            # add val loss to log
            logging.info('Epoch: {}, Val Loss: {}'.format(epoch, loss_val_sum))
            '''