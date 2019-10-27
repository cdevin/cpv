import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from gridworld.algorithms.models import  layer_init
#from gridworld.algorithms.resnet_models import ResNetCNN
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=64, num_outputs=1):
        super().__init__()
        self.fc_1  = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc_2  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.outfc =  layer_init(nn.Linear(hidden_dim, feature_dim))
        
            
    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.reshape(x, (batch_size, -1))
        y = F.relu(self.fc_1(y))
        y = F.relu(self.fc_2(y))
        #y = F.relu(self.fc_3(y))
        output= self.outfc(y)
        return output
    
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=64):
        super().__init__()
        self.fc_1  = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc_2  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_3  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_4  = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.outfc =  layer_init(nn.Linear(hidden_dim, feature_dim))
        #self.out_fc = self.outfc0
            
    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.reshape(x, (batch_size, -1))
        y = F.relu(self.fc_1(y))
        y = F.relu(self.fc_2(y))
        output = self.outfc(y)
        return output
    
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels=4, feature_dim=64, agent_centric = False):
        super().__init__()
        self.feature_dim = feature_dim
        #self.fc_out = out_channels*11*10#*22*20
        if agent_centric:
            self.fc_out = out_channels*22*20#*22*20
        else:
            self.fc_out = out_channels*21*20#*22*20
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=3))
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1))
        self.conv3 = layer_init(nn.Conv2d(32, out_channels, kernel_size=3, stride=1,padding=1))
        #self.conv4 = layer_init(nn.Conv2d(64, out_channels, kernel_size=3, stride=1,padding=1))
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
        return y

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
    
class StateTaskEmbeddingModel(nn.Module):
    def __init__(self, use_init_state=True, use_ref_first_state=True, state_dim=32, task_embedding_dim=20, 
                 action_dim=6, margin=0.1, lambda_ctrl=0.2, lambda_emb=1.0):
        super().__init__()
        self.margin=margin
        self.lambda_ctrl = lambda_ctrl
        self.lambda_emb = lambda_emb
        if use_ref_first_state:
            self.task_embedding = MLP2(state_dim*2, feature_dim=task_embedding_dim)
        else:
            self.task_embedding = MLP2(state_dim, feature_dim=task_embedding_dim)
        if use_init_state:
            self.control = MLP2(task_embedding_dim+state_dim*2, feature_dim=action_dim)
        else:
            self.control = MLP2(task_embedding_dim+state_dim*2, feature_dim=action_dim)
            
    def embed_task(self,ref_final_state, ref_init_state =None, normalize=True):
        if ref_init_state is not None:
            task = torch.cat((ref_init_state, ref_final_state), dim=1)
        else: 
            task = ref_final_state
        sentence = self.task_embedding(task)
        if normalize:
            norms = torch.norm(sentence, p=2, dim=1, keepdim=True)
            sentence = sentence/norms
        return sentence
        
    def forward(self, init_observation, observation, ref_init_state, ref_final_state):
        sentence =self.embed_task(ref_final_state, ref_init_state =ref_init_state)
        if init_observation is not None:
            obs =  torch.cat((observation, init_observation, sentence), dim=1)
        else: 
            obs =  torch.cat((observation, sentence), dim=1)
        actions = self.control(obs)
        return actions, sentence
    
    
    def embedding_loss(self,anchor_s, positive_s, negative_s):
        pos = torch.bmm(anchor_s.unsqueeze(1), positive_s.unsqueeze(2)).squeeze(1)#.squeeze(1)
        #print(pos.shape)
        neg = torch.bmm(anchor_s.unsqueeze(1), negative_s.unsqueeze(2)).squeeze(1)#.squeeze(1)
        #print(neg.shape)
        #batch_size = neg.shape[0]
        compare = torch.cat((torch.torch.zeros_like(pos), self.margin - pos + neg), dim=1)
        #import pdb; pdb.set_trace()
        loss,_ = torch.max(compare, dim=1)
        #print(loss)
        return torch.mean(loss)
    
    def control_loss(self, logpi, label):
        return torch.nn.functional.cross_entropy(logpi, label)
    
    def total_loss(self, embedding_loss, control_loss):
        return self.lambda_ctrl*control_loss + self.lambda_emb*embedding_loss
    
class ImageTaskEmbeddingModel(StateTaskEmbeddingModel):
    def __init__(self, use_init_state=True, use_ref_first_state=True,task_embedding_dim=256, 
                 action_dim=6, margin=0.1, lambda_ctrl=0.2, lambda_emb=1.0):
        nn.Module.__init__(self)
        self.margin=margin
        self.lambda_ctrl = lambda_ctrl
        self.lambda_emb = lambda_emb
        if use_ref_first_state:
            self.task_embedding =  CNN2(6, out_channels=64, feature_dim=task_embedding_dim)
        else:
            self.task_embedding = CNN2(3, out_channels=64, feature_dim=task_embedding_dim)
        if use_init_state:
            self.img_cnn = CNN2(6,out_channels=64, feature_dim=task_embedding_dim)
        else:
            self.img_cnn = CNN2(3, out_channels=64, feature_dim=task_embedding_dim)
        self.control = MLP2(task_embedding_dim*2, feature_dim=action_dim)

        
    def get_goal_feat(self,  ref_init_state, ref_final_state):
        sentence =self.embed_task(ref_final_state, ref_init_state =ref_init_state)
        #goal = torch.cat((pre_image, post_image), dim=1)
        #goal_feat = self.goal_cnn(goal)
        return sentence
        
#     def forward(self, first_image,image, pre_image=None, post_image=None, final_image = None, return_delta=False, goal_feat=None):
#         if goal_feat is None:
#             goal_feat = self.get_goal_feat(pre_image, post_image)
    def forward(self, init_observation, observation, ref_init_state=None, ref_final_state=None, goal_feat=None):
        sentence=goal_feat
        if sentence is None:
            sentence =self.embed_task(ref_final_state, ref_init_state =ref_init_state)
        if init_observation is not None:
            obs =  torch.cat((observation, init_observation), dim=1)
        else: 
            obs =  torch.cat((observation), dim=1)
        obs = self.img_cnn(obs)
        actions = self.control(torch.cat((obs, sentence), dim=1))
        return actions, sentence
    
