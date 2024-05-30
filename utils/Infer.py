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


class BLS_Infer():

    def __init__(self, test_dataset, model, device, batch_size = 1, learning_rate = 0.001, num_epochs = 100, seed = 30032023):
        

        self.test_dataset = test_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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

    def infer(self):

        device = self.device

        # loss
        loss_sum = 0
        loss_total_sum = 0
        loss_hand_pos_sum = 0
        loss_arm_pos_sum = 0
        loss_sholder_pos_sum = 0
        loss_depth_sum = 0
        loss_rate_sum = 0
        loss_release_sum = 0

        # GD score
        total_score_list = []
        hand_pos_score_list = []
        arm_pos_score_list = []
        sholder_pos_score_list = []
        depth_score_list = []
        rate_score_list = []
        release_score_list = []

        # output score
        total_score_output_list = []
        hand_pos_score_output_list = []
        arm_pos_score_output_list = []
        sholder_pos_score_output_list = []
        depth_score_output_list = []
        rate_score_output_list = []
        release_score_output_list = []


        for (i , inputs) in enumerate(self.test_loader):

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
                loss_total = self.criterion(total, total_score)
                loss_hand_pos = self.criterion(hand_pos, hand_pos_score)
                loss_arm_pos = self.criterion(arm_pos, arm_pos_score)
                loss_sholder_pos = self.criterion(sholder_pos, sholder_pos_score)
                loss_depth = self.criterion(depth, depth_score)
                loss_rate = self.criterion(rate, rate_score)
                loss_release = self.criterion(release, release_score)

                # add GD score to list
                total_score_list.append(total_score.item())
                hand_pos_score_list.append(hand_pos_score.item())
                arm_pos_score_list.append(arm_pos_score.item())
                sholder_pos_score_list.append(sholder_pos_score.item())
                depth_score_list.append(depth_score.item())
                rate_score_list.append(rate_score.item())
                release_score_list.append(release_score.item())

                # add output score to list but just keep two decimal
                total_score_output_list.append(round(total.item(), 2))
                hand_pos_score_output_list.append(round(hand_pos.item(), 2))
                arm_pos_score_output_list.append(round(arm_pos.item(), 2))
                sholder_pos_score_output_list.append(round(sholder_pos.item(), 2))
                depth_score_output_list.append(round(depth.item(), 2))
                rate_score_output_list.append(round(rate.item(), 2))
                release_score_output_list.append(round(release.item(), 2))

                # add loss to sum
                loss_sum += loss.item()
                loss_total_sum += loss_total.item()
                loss_hand_pos_sum += loss_hand_pos.item()
                loss_arm_pos_sum += loss_arm_pos.item()
                loss_sholder_pos_sum += loss_sholder_pos.item()
                loss_depth_sum += loss_depth.item()
                loss_rate_sum += loss_rate.item()
                loss_release_sum += loss_release.item()


        #  add loss to tensorboard
        self.writer.add_scalar('Loss/test', loss_sum, 0)
        # save list of GD score
        np.save('output/total_score_list.npy', total_score_list)
        np.save('output/hand_pos_score_list.npy', hand_pos_score_list)
        np.save('output/arm_pos_score_list.npy', arm_pos_score_list)
        np.save('output/sholder_pos_score_list.npy', sholder_pos_score_list)
        np.save('output/depth_score_list.npy', depth_score_list)
        np.save('output/rate_score_list.npy', rate_score_list)
        np.save('output/release_score_list.npy', release_score_list)
        # add all GD score to logger
        logging.info('Score/test_total: {}'.format(total_score_list))
        logging.info('Score/test_hand_pos: {}'.format(hand_pos_score_list))
        logging.info('Score/test_arm_pos: {}'.format(arm_pos_score_list))
        logging.info('Score/test_sholder_pos: {}'.format(sholder_pos_score_list))
        logging.info('Score/test_depth: {}'.format(depth_score_list))
        logging.info('Score/test_rate: {}'.format(rate_score_list))
        logging.info('Score/test_release: {}'.format(release_score_list))
        # save list of output score
        np.save('output/total_score_output_list.npy', total_score_output_list)
        np.save('output/hand_pos_score_output_list.npy', hand_pos_score_output_list)
        np.save('output/arm_pos_score_output_list.npy', arm_pos_score_output_list)
        np.save('output/sholder_pos_score_output_list.npy', sholder_pos_score_output_list)
        np.save('output/depth_score_output_list.npy', depth_score_output_list)
        np.save('output/rate_score_output_list.npy', rate_score_output_list)
        np.save('output/release_score_output_list.npy', release_score_output_list)
        # add all output score to logger
        logging.info('Score/test_total_output: {}'.format(total_score_output_list))
        logging.info('Score/test_hand_pos_output: {}'.format(hand_pos_score_output_list))
        logging.info('Score/test_arm_pos_output: {}'.format(arm_pos_score_output_list))
        logging.info('Score/test_sholder_pos_output: {}'.format(sholder_pos_score_output_list))
        logging.info('Score/test_depth_output: {}'.format(depth_score_output_list))
        logging.info('Score/test_rate_output: {}'.format(rate_score_output_list))
        logging.info('Score/test_release_output: {}'.format(release_score_output_list))

        # compute mae between GD score and output score
        loss_test = np.mean(np.abs(np.array(total_score_list) - np.array(total_score_output_list)))
        loss_total_test = np.mean(np.abs(np.array(total_score_list) - np.array(total_score_output_list)))
        loss_hand_pos_test = np.mean(np.abs(np.array(hand_pos_score_list) - np.array(hand_pos_score_output_list)))
        loss_arm_pos_test = np.mean(np.abs(np.array(arm_pos_score_list) - np.array(arm_pos_score_output_list)))
        loss_sholder_pos_test = np.mean(np.abs(np.array(sholder_pos_score_list) - np.array(sholder_pos_score_output_list)))
        loss_depth_test = np.mean(np.abs(np.array(depth_score_list) - np.array(depth_score_output_list)))
        loss_rate_test = np.mean(np.abs(np.array(rate_score_list) - np.array(rate_score_output_list)))
        loss_release_test = np.mean(np.abs(np.array(release_score_list) - np.array(release_score_output_list)))
        

        # add all loss to logger
        logging.info('Loss/test: {}'.format(loss_test))
        logging.info('Loss/test_total: {}'.format(loss_total_test))
        logging.info('Loss/test_hand_pos: {}'.format(loss_hand_pos_test))
        logging.info('Loss/test_arm_pos: {}'.format(loss_arm_pos_test))
        logging.info('Loss/test_sholder_pos: {}'.format(loss_sholder_pos_test))
        logging.info('Loss/test_depth: {}'.format(loss_depth_test))
        logging.info('Loss/test_rate: {}'.format(loss_rate_test))
        logging.info('Loss/test_release: {}'.format(loss_release_test))
        

        return loss_test, loss_total_test, loss_hand_pos_test, loss_arm_pos_test, loss_sholder_pos_test, loss_depth_test, loss_rate_test, loss_release_test

