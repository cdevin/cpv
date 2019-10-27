
from __future__ import print_function
import time
import cv2
import argparse
import os
import torch
import time
#import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from gridworld.algorithms.composite_dataset import CompositeDataset
import numpy as np
import copy
from gridworld.algorithms.composite_models import CompositeDotModelV3
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train_size', type=int, default=None,
                    help='which task to run')
parser.add_argument('--test_size', type=int, default=1000,
                    help='which task to run')
parser.add_argument('--resume', type=str, default=None,
                    help='A model to load')
parser.add_argument('--modelV', type=int, default=3,
                    help='A model to load')
parser.add_argument('--feature-dim', type=int, default=256)a
parser.add_argument('-H', action='store_true', default=False,
                    help='use hom loss')
parser.add_argument('-P', action='store_true', default=False,
                    help='use pair loss')
parser.add_argument('--letter', type=str, default='A',
                    help='Suffix to model weights')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}
print("train_loader")

data_dir = '/home/coline/affordance_world/data/Oct_4tasks_images/'

    
def worker_init_fn(worker_id):      
    np.random.seed()
    st0 = np.random.get_state()[1][0]

train_loader = torch.utils.data.DataLoader(
    CompositeDataset(directory=data_dir,
                             train=True, size=args.train_size),
    batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
print("test_loader")
test_loader = torch.utils.data.DataLoader(
    CompositeDataset(directory=data_dir,
                             train=False, size=args.test_size),
    batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
args.train_size = len(train_loader.dataset)
start_time = time.time()
result_list = []

model = CompositeDotModelV3(device,task_embedding_dim=args.feature_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("done model")

def accuracy(logpi, label):
    max_index = logpi.max(dim = 1)[1]
    #import pdb; pdb.set_trace()
    return (max_index == label).sum()

def v3_loss_function(logpi, label):
    return torch.nn.functional.cross_entropy(logpi, label)

def g_matching_loss(ref_g, exp_g):
    loss = torch.nn.functional.triplet_margin_loss(ref_g[:-1], exp_g[:-1], exp_g[1:], margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='elementwise_mean')
    return loss

def process_data(data):
    image = data['image'].to(device)
    post_image = data['post_image'].to(device)
    pre_image = data['pre_image'].to(device)
    first_image = data['init_image'].to(device)
    final_image = data['final_image'].to(device)
    action =  data['action'].to(device) 
 
    return first_image, image, final_image,  pre_image, post_image, action#, (pos_x, pos_y)

def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    num_batch = 0
    feat_loss_total = 0
    for batch_idx, data in enumerate(train_loader):
        first_image, image,final_image, pre_image, post_image, action = process_data(data)
        optimizer.zero_grad()
        logpi, goal_feat, final_feat, back_feat =model.forward(first_image, image, pre_image, post_image, final_image=final_image)
        forward_feat = model.get_goal_feat(image, final_image)
        loss = v3_loss_function(logpi, action)
        total_feat_loss = 0
        if args.P:
            feat_loss = g_matching_loss(goal_feat, final_feat)
            loss+= feat_loss#+feat_loss2
            total_feat_loss += feat_loss.item()
        if args.H:
            feat_loss = g_matching_loss(forward_feat+back_feat, final_feat)
            loss+= feat_loss#+feat_loss2
            total_feat_loss += feat_loss.item()

        acc = accuracy(logpi,action).item() #/len(image)
        num_batch +=len(image)
        loss.backward()
        train_loss += loss.item() #/len(image)
        train_acc += acc#.item()# /len(image)
        
        optimizer.step()
        #prev_image, prev_last_image, prev_pos = image, last_image, pos
        #prev_deltas, prev_features = deltas, agent_features
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() ,
            ),
            )
        #print("time", time.time()-start_time)
    writer.add_scalar('Accuracy/train', train_acc/num_batch, epoch )
    writer.add_scalar('Loss/train',train_loss / num_batch, epoch )
    writer.add_scalar('Regloss/train',total_feat_loss/num_batch, epoch)

    print('====> Epoch: {} Average loss: {:.4f} Accuracy: {:.4f} FeatLoss: {:.4f} '.format(
          epoch, train_loss / num_batch, train_acc/num_batch, feat_loss_total/num_batch))
    
def test(epoch):
    model.eval()
    test_loss = 0
    test_acc = 0
    num_batch = 0
    total_feat_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            first_image, image, final_image,  pre_image, post_image, action = process_data(data)
            logpi, goal_feat, final_feat, back_feat =model.forward(first_image, image, pre_image, post_image, final_image=final_image)
            forward_feat = model.get_goal_feat(image, final_image)
            loss = v3_loss_function(logpi, action)
            if args.P:
                feat_loss = g_matching_loss(goal_feat, final_feat)
                loss+= feat_loss#+feat_loss2
                total_feat_loss += feat_loss.item()
            if args.H:
                feat_loss = g_matching_loss(forward_feat+back_feat, final_feat)
                loss+= feat_loss
                total_feat_loss += feat_loss.item()

            acc = accuracy(logpi,action).item() 
            test_loss += loss.item()
            test_acc += acc
            num_batch += len(image)
            
    writer.add_scalar('Loss/test', test_loss/num_batch, epoch )
    writer.add_scalar('Regloss/test',total_feat_loss/num_batch, epoch)
    writer.add_scalar('Accuracy/test', test_acc/num_batch, epoch )
    print('====> Test set loss: {:.4f} Accuracy: {:.4f} FeatLoss: {:.4f} '.format(test_loss/num_batch, test_acc/num_batch, total_feat_loss/num_batch))

def save_model(model, epoch, path):
    state_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    torch.save((state_dict, epoch, len(train_loader.dataset), opt_dict), path)

hl_str = ''
pl_str = ''
if args.H:
    hl_str ='_hom'
if args.P:
    pl_str ='_pair'
if __name__ == "__main__":
    filename = 'subtraction_4tasks'+hl_str+pl_str+'_images_feat'+str(args.feature_dim)+'_'+args.letter+'_{:06d}'.format(args.train_size)
    writer = SummaryWriter(comment=filename)
    basedirectory = 'octresults/'+filename
    print("Tensorbaord logdir",  writer.logdir)
    if not os.path.isdir(basedirectory):
        os.mkdir(basedirectory)
    if args.resume is not None:
        model_path = args.resume
        #start_epoch = int(model_path[-5:-3])
        state_dict, start_epoch, _, opt_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(opt_dict)
    else:
        start_epoch = 0
        test(0)
        path_name =  basedirectory+'/epoch{:05d}.pt'.format( 0)
        print('saved',path_name)
        save_model(model, 0, path_name)
    #test(0)
    for epoch in range(start_epoch+1, args.epochs + 1):
        print("starting train")
        t = time.time()
        writer.add_scalar("Time/time", t, epoch)
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            path_name =  basedirectory+'/epoch{:05d}.pt'.format(epoch)
            print('saved',path_name)
            save_model(model, epoch,  path_name)
        t2 = time.time()
        writer.add_scalar("Time/delta", t2-t, epoch)