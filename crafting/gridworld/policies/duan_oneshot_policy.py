import numpy as np
import math
import random

from rlkit.policies.base import SerializablePolicy
import torch
import glob
#from skimage import io, transform
import cv2
from torchvision import transforms, utils
from gridworld.envs.grid_affordance import HammerWorld, ACTIONS
from gridworld.algorithms.duan_oneshot_models import DuanAttentionModel
import copy
from torch.nn import functional as F
from gridworld.algorithms.duan_oneshot_dataset import ActionToTensor
# 0: UP
import torch

class LearnedDuanAttentionPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=DuanAttentionModel, agent_centric=False, env_dim=(63,60,3), task_embedding_dim=256,relu=False):
        print("model", model, task_embedding_dim)
        self.model = model(device).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
        
    
    def get_action(self, curr_traj, ref_traj):
        with torch.no_grad():
            #img = img.reshape(self.env_dim)#*255
            #import pdb; pdb.set_trace()
            ref_traj = [obs for obs in ref_traj]
            ref_len = torch.tensor([len(ref_traj)]).to(self.device)
            curr_len = torch.tensor([len(curr_traj)]).to(self.device)
            max_len_string = max(len(ref_traj), len(curr_traj))
            if len(ref_traj) < max_len_string:
                padding = [ref_traj[-1]]*(max_len_string-len(ref_traj))
                ref_traj = ref_traj+padding
            if len(curr_traj) < max_len_string:
                padding = [curr_traj[-1]]*(max_len_string-len(curr_traj))
                curr_traj = curr_traj+padding
            #print(len(ref_traj), len(curr_traj), max_len_string)
            ref = torch.stack([self.transformer.convert_image(i.reshape(self.env_dim)) for i in ref_traj]).unsqueeze(0).to(self.device)
            curr = torch.stack([self.transformer.convert_image(i.reshape(self.env_dim)) for i in curr_traj]).unsqueeze(0).to(self.device)
            
            logits = self.model.forward(curr,curr_len, ref , ref_len)#[0].squeeze(0).squeeze(0)
            logits = logits[curr_len-1]
            #import pdb; pdb.set_trace()
            M = torch.distributions.categorical.Categorical( logits=logits)
            action = M.sample().item()
        return action
