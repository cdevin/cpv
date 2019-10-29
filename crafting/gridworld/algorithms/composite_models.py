import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from gridworld.algorithms.models import layer_init

    
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=64, num_outputs=1):
        super().__init__()
        self.fc_1  = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc_2  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_3  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_4  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.outfc0 =  layer_init(nn.Linear(hidden_dim, feature_dim))
        self.out_fcs = [self.outfc0]
        if num_outputs >1:
            for i in range(1, num_outputs):
                fc = layer_init(nn.Linear(hidden_dim, feature_dim))
                self.__setattr__("outfc{}".format(i), fc)
                self.out_fcs.append(fc)
            
    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.reshape(x, (batch_size, -1))
        y = F.relu(self.fc_1(y))
        y = F.relu(self.fc_2(y))
        #y = F.relu(self.fc_3(y))
        outputs = []
        for fc in self.out_fcs:
            h = fc(y).unsqueeze(1)
            outputs.append(h)

        return outputs
    

class CNN2(nn.Module):
    def __init__(self, in_channels, out_channels=4, feature_dim=64, agent_centric = False):
        super().__init__()
        self.feature_dim = feature_dim
        #self.fc_out = out_channels*11*10#*22*20
        if agent_centric:
            self.fc_out = out_channels*22*20#*22*20
        else:
            self.fc_out = out_channels*21*20#*22*20
        self.fc_out = out_channels*3*3
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=3))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=5, stride=2,padding=2))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1))
        self.conv4 = layer_init(nn.Conv2d(64, out_channels, kernel_size=3, stride=1,padding=1))
        #self.conv5 = layer_init(nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1))
        #self.conv6 = layer_init(nn.Conv2d(256, 128, kernel_size=3, stride=1,padding=1))
        #self.conv7 = layer_init(nn.Conv2d(128, out_channels, kernel_size=3, stride=1,padding=1))
        self.fc_1  = layer_init(nn.Linear(self.fc_out, feature_dim))
        self.convs = [self.conv1, self.conv2, self.conv3]#, self.conv4,  self.conv7]


    def forward(self, x):
        batch_size = x.shape[0]
        y = self.conv1(x)
        for k in range(1,len(self.convs)):
            y = F.relu(y)
            y = self.convs[k](y)
            #print(y.shape)
        y = torch.reshape(y, (batch_size, -1))
        #print(y.shape)
        y = F.relu(y)
        y = self.fc_1(y)
        #print(y.shape)
        return y
    


class CompositeDotModelV3(nn.Module):
    """ [f(s_0, a), (g(s*_0, s*_T) -g(s_0, s_t))]"""
    def __init__(self, device=None,task_embedding_dim=256, relu=False):
        super().__init__()
        self.goal_cnn = CNN2(6, out_channels=64, feature_dim=task_embedding_dim)
        self.image_cnn = CNN2(3, out_channels= 64, feature_dim=task_embedding_dim)
        self.mlp = MLP2(input_dim=task_embedding_dim*2, hidden_dim=256, feature_dim=6, num_outputs=6)
        self.relu = relu
    
    def get_goal_feat(self, pre_image, post_image):
        goal = torch.cat((pre_image, post_image), dim=1)
        goal_feat = self.goal_cnn(goal)
        if self.relu:
            goal_feat = F.relu(goal_feat)
        return goal_feat
        
    def forward(self, first_image,image, pre_image=None, post_image=None, final_image = None, return_delta=False, goal_feat=None):
        if goal_feat is None:
            goal_feat = self.get_goal_feat(pre_image, post_image)
        back_goal = self.get_goal_feat(first_image, image)
        
        img_feat = self.image_cnn(image)
        obs = torch.cat((img_feat,  goal_feat-back_goal), dim=1)
        actions = self.mlp(obs)[0].squeeze(1)
        if final_image is not None:
            final_features = self.get_goal_feat(first_image, final_image)
            return actions, goal_feat, final_features, back_goal
        if return_delta:
            return actions, goal_feat-back_goal
        return actions
    
class TaskEmbeddingModel(nn.Module):
    """ [f(s_0, a), (g(s*_0, s*_T) -g(s_0, s_t))]"""
    def __init__(self, device=None,task_embedding_dim=256, relu=False):
        super().__init__()
        self.goal_cnn = CNN2(6, out_channels=64, feature_dim=task_embedding_dim)
        self.image_cnn = CNN2(3, out_channels= 64, feature_dim=task_embedding_dim)
        self.mlp = MLP2(input_dim=task_embedding_dim*3, hidden_dim=256, feature_dim=6, num_outputs=6)
        self.relu = relu
    
    def get_goal_feat(self, pre_image, post_image):
        goal = torch.cat((pre_image, post_image), dim=1)
        goal_feat = self.goal_cnn(goal)
        if self.relu:
            goal_feat = F.relu(goal_feat)
        return goal_feat
        
    def forward(self, first_image,image, pre_image=None, post_image=None, final_image = None, return_delta=False, goal_feat=None):
        if goal_feat is None:
            goal_feat = self.get_goal_feat(pre_image, post_image)
        back_goal = self.get_goal_feat(first_image, image)
        img_feat = self.image_cnn(image)
        first_image_feat = self.image_cnn(first_image)
        obs = torch.cat((img_feat, first_image_feat, goal_feat), dim=1)
        actions = self.mlp(obs)[0].squeeze(1)
        if final_image is not None:
            final_features = self.get_goal_feat(first_image, final_image)
            return actions, goal_feat, final_features, back_goal
        if return_delta:
            return actions, goal_feat-back_goal
        return actions
    


    
class NaiveModel(nn.Module):
    """ [f(s_0, a), (g(s*_0, s*_T) -g(s_0, s_t))]"""
    def __init__(self, device=None,task_embedding_dim=256, relu=False):
        super().__init__()
        self.image_cnn = CNN2(12, out_channels= 64, feature_dim=task_embedding_dim)
        self.mlp = MLP2(input_dim=task_embedding_dim, hidden_dim=256, feature_dim=6, num_outputs=6)
        
    def forward(self, first_image,image, pre_image, post_image):
        obs = torch.cat((pre_image, post_image, first_image, image), dim=1)
        obs = self.image_cnn(obs)
        actions = self.mlp(obs)[0].squeeze(1)
        return actions
    
