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
#from gridworld.algorithms.composite_models import CompositeDotModel, CompositeBCModel#, ActionDeltaModel, StateActionDeltaModel
import copy
from torch.nn import functional as F
from gridworld.algorithms.composite_dataset import ActionToTensor, StateActionToTensor
# 0: UP
import torch
class LearnedDeltaPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=None, agent_centric=False,  env_dim=(63,60,3),relu=False):
        print("model", model)
        self.model = model(device).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
        
    def action_classification(self, deltas, features):
        """
        deltas is batch x feature_dim
        features is batch x action x feature_dim
        labels is batch x 1
        """
        deltas = deltas.unsqueeze(2)
        batch, nA, nF = features.shape

        classification = torch.bmm(features, deltas).squeeze(2)[0]
        M = torch.distributions.categorical.Categorical( logits=classification)
        action = M.sample().item()
        return action
        
    def get_delta_star(self, img_pre, img_post):
        #import pdb; pdb.set_trace()
        img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
        img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
        goal = torch.cat((img_pre, img_post), dim=1)
        goal_feat = self.model.goal_cnn(goal)

        return goal_feat
    
    def get_action(self,  img, img_pre, img_post, first_image ):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape((66,60,3))#*255
            else:
                img = img.reshape((63,60,3))#*255
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
            img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
            if first_image is not None:
                img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
                deltas, features  = self.model.forward(img_first, image, img_pre, img_post)#.squeeze(0).squeeze(0)
            else:
                deltas, features  = self.model.forward(image, img_pre, img_post)#.squeeze(0).squeeze(0)
            #features = self.model.mlp(self.model.image_cnn(batch_current))
            deltas = deltas.squeeze(1)
            features = torch.cat(features, dim=1)
            batch_size, num_actions, feature_dim = features.shape
            deltas = deltas.unsqueeze(2)
            batch, nA, nF = features.shape

            logits = torch.bmm(features, deltas).squeeze(2)
            M = torch.distributions.categorical.Categorical( logits=logits)
            
            #action = torch.argmax(logits).item()
            action = M.sample().item()
            #action = self.action_classification(deltas, features)
            
        return action
    
    
class LearnedBCPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=None, agent_centric=False, env_dim=(63,60,3), task_embedding_dim=256,relu=False):
        print("model", model)
        self.model = model(task_embedding_dim=task_embedding_dim, relu=relu).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
        
    def get_goal_feat(self, img_pre, img_post):
        img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
        img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
        goal_feat = self.model.get_goal_feat(img_pre, img_post)
        return goal_feat
        
    def get_action_from_ref(self, img, first_image, goal_feat,return_delta=False):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)#*255
            else:
                img = img.reshape(self.env_dim)#*255
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
            if return_delta:
                logits= self.model.forward(img_first, image,goal_feat=goal_feat, return_delta=return_delta)
            else:
                logits = self.model.forward(img_first, image,goal_feat=goal_feat)
            if return_delta:
                logits, delta = logits
            logits = logits.squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical( logits=logits)
            action = M.sample().item()
        if return_delta:
            return action, delta
        return action
    
    def get_action(self, img, img_pre, img_post, first_image = None, return_delta=False):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)#*255
            else:
                img = img.reshape(self.env_dim)#*255
            #import pdb; pdb.set_trace()
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
            img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
            if first_image is not None:
                img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
                if return_delta:
                    logits= self.model.forward(img_first, image, img_pre, img_post, return_delta=return_delta)
                else:
                    logits = self.model.forward(img_first, image, img_pre, img_post)
            else:
                logits = self.model.forward(image, img_pre, img_post,return_delta=return_delta)
            if return_delta:
                logits, delta = logits
            logits = logits.squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical( logits=logits)
            action = M.sample().item()
        if return_delta:
            return action, delta
        return action
    
class LearnedStateBCPolicy(LearnedBCPolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=None, agent_centric=False, env_dim=(32)):

        print("model", model)
        self.model = model().to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = StateActionToTensor()
        self.device = device
        self.env_dim= env_dim
    

    
    
class LearnedSingleBCPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=None, agent_centric=False,  env_dim=(63,60,3), relu=False):
        print("model", model)
        self.model = model().to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
    
    def get_action(self, img, img_pre, img_post, first_image = None):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)#*255
            else:
                img = img.reshape(self.env_dim)#*255
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            logits = self.model.forward(image).squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical( logits=logits)
            action = M.sample().item()
        return action
    
class LearnedLabeledBCPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model, agent_centric=False,  env_dim=(63,60,3),relu=False):
        print("model", model)
        self.model = model(device=device).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
    
    def get_action(self, img, first_image, task):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)#*255
            else:
                img = img.reshape(self.env_dim)#*255
            #import pdb; pdb.set_trace()
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            task = torch.tensor([task]).to(self.device)
            img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
            logits = self.model.forward(img_first, image, task).squeeze(0).squeeze(0)
            M = torch.distributions.categorical.Categorical( logits=logits)
            #import pdb; pdb.set_trace()
            action = M.sample().item()
        return action
    
    
    
class LearnedMiddlesPolicy(SerializablePolicy):
    """
    
    """

    def __init__(self,  state_dict,device, model=None, agent_centric=False, env_dim=(63,60,3),relu=False):
        print("model", model)
        self.model = model(device).to(device)
        self.model.load_state_dict(state_dict)
        self.agent_centric=agent_centric
        self.model.eval()
        self.transformer = ActionToTensor()
        self.device = device
        self.env_dim= env_dim
    
    def get_action(self, img, img_pre, img_post, first_image, ref_middles, curr_middles):
        with torch.no_grad():
            if self.agent_centric:
                img = img.reshape(self.env_dim)#*255
            else:
                img = img.reshape(self.env_dim)#*255
            #import pdb; pdb.set_trace()
            image = self.transformer.convert_image(img).unsqueeze(0).to(self.device)
            img_pre = self.transformer.convert_image(img_pre).unsqueeze(0).to(self.device)
            img_post = self.transformer.convert_image(img_post).unsqueeze(0).to(self.device)
            img_first = self.transformer.convert_image(first_image).unsqueeze(0).to(self.device)
            curr_middles=  torch.stack([self.transformer.convert_image(i) for i in curr_middles]).unsqueeze(0).to(self.device)
            ref_middles=  torch.stack([self.transformer.convert_image(i) for i in ref_middles]).unsqueeze(0).to(self.device)
            logits = self.model.forward(img_first, image, img_pre, img_post, curr_middles, ref_middles).squeeze(0).squeeze(0)

            M = torch.distributions.categorical.Categorical( logits=logits)
            action = M.sample().item()
        return action