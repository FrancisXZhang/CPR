import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from utils.BLSLoader import BLSRatingsDataset
from utils.Train import BLS_Train
from utils.Infer import BLS_Infer
from net.STGCN import MutltiViewModel as Model

from datetime import datetime
import logging

import numpy as np
import pandas as pd
from natsort import natsorted

from sklearn.model_selection import train_test_split, KFold
import wandb

class main_proc():

    def __init__(self):

        super(main_proc, self).__init__()

    def main_proc(self, config=None):

        with wandb.init(config=config):
            config = wandb.config
            # set logger
            now = datetime.now()
            current_folder_name = os.path.basename(os.getcwd())
            log_name = current_folder_name + '_' + now.strftime("%m_%d_%Y_%H_%M_%S") + '.log'
            logging.basicConfig( filename=log_name,level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            
            data_dir = ''
            joint_folder = '/home2/slxb76/CPR_AQA/MediaPipe/pose_tensor'

            joint_dir = os.path.join(data_dir, joint_folder)
            # get the label
            table_file = 'agreement_clean.csv'
            print(table_file)
            label = pd.read_csv(table_file)
            label = label


            # get the joint files
            joint_files = natsorted(glob.glob(os.path.join(joint_dir, '*.pt')))
            # set camera
            camera_list_candidate = [[0], [0, 1, 2], [0, 4, 5],
                                    [3], [3, 1, 2], [3, 4, 5],]
            camera_list = camera_list_candidate[config.camera_list]


            # Split the data into 60% training, 20% validation, and 20% test in folds
            fold_n = 5
            kf = KFold(n_splits=fold_n, shuffle=True, random_state=42)
            train_label_list = []
            test_label_list = []
            train_joints_list = []
            test_joints_list = []
            for train_index, test_index in kf.split(label):

                train_label, test_label = label.iloc[train_index], label.iloc[test_index]

                train_label_list.append(train_label)
                test_label_list.append(test_label)

                # get the train, val, test joints list
                train_joints = []
                val_joints = []
                test_joints = []
                for train_id in train_index:
                    train_joints.append(joint_files[train_id])
                for test_id in test_index:
                    test_joints.append(joint_files[test_id])

                train_joints_list.append(train_joints)
                test_joints_list.append(test_joints)

            loss_test_list = []
            loss_total_test_list = []
            loss_hand_pos_test_list = []
            loss_arm_pos_test_list = []
            loss_sholder_pos_test_list = []
            loss_depth_test_list = []
            loss_rate_test_list = []
            loss_release_test_list = []

            for i in range(fold_n):

                # train
                print('Training...')

                train_label = train_label_list[i]
                test_label = test_label_list[i]

                train_joints = train_joints_list[i]
                test_joints = test_joints_list[i]        
                train_dataset = BLSRatingsDataset(train_joints, train_label)

                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Model(camera_list, 4, 1, graph_args={'max_hop': 1, 'strategy': 'uniform', 'dilation': 1},
                            edge_importance_weighting= False, attention= False).to(device)

                train = BLS_Train(train_dataset, model, device, num_epochs=config.num_epochs, 
                                learning_rate=config.learning_rate,
                                weight_decay=config.weight_decay,)
                
                train.train()


                # test
                print('Testing...')
                # load model
                model.load_state_dict(torch.load('model.pth'))
                # test for antony conner
                # start log for antony conner

                test_dataset = BLSRatingsDataset(test_joints, test_label)
                test = BLS_Infer(test_dataset, model, device)
                loss_test, loss_total_test, loss_hand_pos_test, loss_arm_pos_test, loss_sholder_pos_test, loss_depth_test, loss_rate_test, loss_release_test = test.infer()
            
                loss_test_list.append(loss_test)
                loss_total_test_list.append(loss_total_test)
                loss_hand_pos_test_list.append(loss_hand_pos_test)
                loss_arm_pos_test_list.append(loss_arm_pos_test)
                loss_sholder_pos_test_list.append(loss_sholder_pos_test)
                loss_depth_test_list.append(loss_depth_test)
                loss_rate_test_list.append(loss_rate_test)
                loss_release_test_list.append(loss_release_test)
            
            # add to wandb log
            wandb.log({'loss_test': np.mean(loss_test_list),
                        'loss_total_test': np.mean(loss_total_test_list),
                        'loss_hand_pos_test': np.mean(loss_hand_pos_test_list),
                        'loss_arm_pos_test': np.mean(loss_arm_pos_test_list),
                        'loss_sholder_pos_test': np.mean(loss_sholder_pos_test_list),
                        'loss_depth_test': np.mean(loss_depth_test_list),
                        'loss_rate_test': np.mean(loss_rate_test_list),
                        'loss_release_test': np.mean(loss_release_test_list)})

