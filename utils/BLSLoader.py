import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from natsort import natsorted
import os
import glob

class BLSRatingsDataset(Dataset):

    def __init__(self, joint_tensors, table_file):


        self.label = table_file

        self.joints = joint_tensors

    def __len__(self):

        return len(self.label)
    
    def __getitem__(self, idx):

        # get the joint
        joint_id = self.joints[idx]
        joint = torch.load(joint_id) 
        max_len = max(joint['view1'].shape[0], joint['view2'].shape[0], joint['view3'].shape[0], joint['view4'].shape[0], joint['view5'].shape[0], joint['view6'].shape[0])
        joint_all = [
            self.pad_tensor(joint['view1'].float(), max_len),
            self.pad_tensor(joint['view2'].float(), max_len),
            self.pad_tensor(joint['view3'].float(), max_len),
            self.pad_tensor(joint['view4'].float(), max_len),
            self.pad_tensor(joint['view5'].float(), max_len),
            self.pad_tensor(joint['view6'].float(), max_len)
        ]
        joint_all = torch.stack(joint_all, dim=0)
        # get the label
        label = self.label.iloc[idx, :]
        total_score = label['Total_points']
        C1_Hand_Pos = label['C1_Hand_Pos']
        C1_Arm_Pos = label['C1_Arm_Pos']
        C1_Shoulder_Pos = label['C1_Shoulder_Pos']
        C1_Depth = label['C1_Depth']
        C1_Rate = label['C1_Rate']
        C1_Release = label['C1_Release']
        C2_Hand_Pos = label['C2_Hand_Pos']
        C2_Arm_Pos = label['C2_Arm_Pos']
        C2_Shoulder_Pos = label['C2_Shoulder_Pos']
        C2_Depth = label['C2_Depth']
        C2_Rate = label['C2_Rate']
        C2_Release = label['C2_Release']
        C3_Hand_Pos = label['C3_Hand_Pos']
        C3_Arm_Pos = label['C3_Arm_Pos']
        C3_Shoulder_Pos = label['C3_Shoulder_Pos']
        C3_Depth = label['C3_Depth']
        C3_Rate = label['C3_Rate']
        C3_Release = label['C3_Release']
        C4_Hand_Pos = label['C4_Hand_Pos']
        C4_Arm_Pos = label['C4_Arm_Pos']
        C4_Shoulder_Pos = label['C4_Shoulder_Pos']
        C4_Depth = label['C4_Depth']
        C4_Rate = label['C4_Rate']
        C4_Release = label['C4_Release']


        sample = {'joint_view1': joint['view1'].float(),
                'joint_view2': joint['view2'].float(),
                'joint_view3': joint['view3'].float(),
                'joint_view4': joint['view4'].float(),
                'joint_view5': joint['view5'].float(),
                'joint_view6': joint['view6'].float(),
                'joint_all': joint_all,
                'total_score': torch.tensor(total_score).float(),
                'C1_Hand_Pos': torch.tensor(C1_Hand_Pos).float(),
                'C1_Arm_Pos': torch.tensor(C1_Arm_Pos).float(),
                'C1_Shoulder_Pos': torch.tensor(C1_Shoulder_Pos).float(),
                'C1_Depth': torch.tensor(C1_Depth).float(),
                'C1_Rate': torch.tensor(C1_Rate).float(),
                'C1_Release': torch.tensor(C1_Release).float(),
                'C2_Hand_Pos': torch.tensor(C2_Hand_Pos).float(),
                'C2_Arm_Pos': torch.tensor(C2_Arm_Pos).float(),
                'C2_Shoulder_Pos': torch.tensor(C2_Shoulder_Pos).float(),
                'C2_Depth': torch.tensor(C2_Depth).float(),
                'C2_Rate': torch.tensor(C2_Rate).float(),
                'C2_Release': torch.tensor(C2_Release).float(),
                'C3_Hand_Pos': torch.tensor(C3_Hand_Pos).float(),
                'C3_Arm_Pos': torch.tensor(C3_Arm_Pos).float(),
                'C3_Shoulder_Pos': torch.tensor(C3_Shoulder_Pos).float(),
                'C3_Depth': torch.tensor(C3_Depth).float(),
                'C3_Rate': torch.tensor(C3_Rate).float(),
                'C3_Release': torch.tensor(C3_Release).float(),
                'C4_Hand_Pos': torch.tensor(C4_Hand_Pos).float(),
                'C4_Arm_Pos': torch.tensor(C4_Arm_Pos).float(),
                'C4_Shoulder_Pos': torch.tensor(C4_Shoulder_Pos).float(),
                'C4_Depth': torch.tensor(C4_Depth).float(),
                'C4_Rate': torch.tensor(C4_Rate).float(),
                'C4_Release': torch.tensor(C4_Release).float(),
                'Hand_Pos': torch.tensor(C1_Hand_Pos + C2_Hand_Pos + C3_Hand_Pos + C4_Hand_Pos).float(),
                'Arm_Pos': torch.tensor(C1_Arm_Pos + C2_Arm_Pos + C3_Arm_Pos + C4_Arm_Pos).float(),
                'Shoulder_Pos': torch.tensor(C1_Shoulder_Pos + C2_Shoulder_Pos + C3_Shoulder_Pos + C4_Shoulder_Pos).float(),
                'Depth': torch.tensor(C1_Depth + C2_Depth + C3_Depth + C4_Depth).float(),
                'Rate': torch.tensor(C1_Rate + C2_Rate + C3_Rate + C4_Rate).float(),
                'Release': torch.tensor(C1_Release + C2_Release + C3_Release + C4_Release).float(),
                }    

        return sample   

    def pad_tensor(self, t, max_len):
        padding_size = max_len - t.shape[0]
        return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, padding_size))