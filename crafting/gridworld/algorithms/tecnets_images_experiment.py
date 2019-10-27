
from __future__ import print_function
import time

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from gridworld.algorithms.composite_dataset import CompositeDataset
import numpy as np
import copy
from tensorboardX import SummaryWriter
import time
import os
from gridworld.algorithms.task_embeddings_networks import ImageTaskEmbeddingModel

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
parser.add_argument('--feature-dim', type=int, default=256)
parser.add_argument('--lambda-ctrl', type=float, default=0.2)
parser.add_argument('-G', action='store_true', default=False,
                    help='on GCP')
parser.add_argument('--avoidant',action='store_true', default=False,
                    help='use avoidant dataset')
parser.add_argument('--resnet',action='store_true', default=False,
                    help='use resnet model')
parser.add_argument('--letter', type=str, default='A',
                    help='A model to load')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
def worker_init_fn(worker_id):      
    np.random.seed()
    st0 = np.random.get_state()[1][0]
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}
data_dir =  '/home/coline/affordance_world/data/Oct_4tasks_images/'

train_loader = torch.utils.data.DataLoader(
    CompositeDataset(directory=data_dir,
                             train=True, size=args.train_size),
    batch_size=args.batch_size, shuffle=True,worker_init_fn=worker_init_fn, **kwargs)
print("test_loader")
test_loader = torch.utils.data.DataLoader(
    CompositeDataset(directory=data_dir,
                             train=False, size=args.test_size),
    batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
print("done laoding")
args.train_size = len(train_loader.dataset)
start_time = time.time()
result_list = []
print("Making model")
if args.resnet:
    model = ResNetImageTaskEmbeddingModel(lambda_ctrl=args.lambda_ctrl, task_embedding_dim=args.feature_dim).to(device)
else:
    model = ImageTaskEmbeddingModel(lambda_ctrl=args.lambda_ctrl, task_embedding_dim=args.feature_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("done model")

def accuracy(logpi, label):
    max_index = logpi.max(dim = 1)[1]
    #import pdb; pdb.set_trace()
    return (max_index == label).sum()

def process_data(data):
    observation = data['image'].to(device)
    ref_final_state = data['post_image'].to(device)
    ref_init_state = data['pre_image'].to(device)
    init_observation = data['init_image'].to(device)
    final_observation = data['final_image'].to(device)
    action =  data['action'].to(device) 
    optimizer.zero_grad()
    return init_observation, observation, final_observation, ref_init_state, ref_final_state , action

def train(epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    num_batches =0
    feat_loss_total = 0
    t1 = time.clock()
    num_obs = 0
    for batch_idx, data in enumerate(train_loader):
        init_observation, observation, final_observation, ref_init_state, ref_final_state, action = process_data(data)
        logpi, anchor_sentence = model(init_observation, observation, ref_init_state, ref_final_state)
        ctrl_loss = model.control_loss(logpi,action)
        positive_sentence = model.embed_task(init_observation, final_observation)
        embedding_loss = model.embedding_loss(anchor_sentence[1:], positive_sentence[1:], positive_sentence[:-1])
        loss = model.total_loss(embedding_loss, ctrl_loss)
        loss.backward()
        train_loss += loss.item() #/len(image)
        acc = accuracy(logpi,action).item() #/len(image)
        train_acc += acc
        optimizer.step()
        num_batches += 1
        feat_loss_total += embedding_loss.item()
        num_obs += len(observation)
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(observation), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(observation),
            ),
            )
    
    writer.add_scalar('Accuracy/train', train_acc/num_obs, epoch )
    writer.add_scalar('Loss/train',train_loss / num_obs, epoch )
    writer.add_scalar('Regloss/train',feat_loss_total/num_obs, epoch)
    #print("time", t2-t1, "for ", num_batches, "batches of size", args.batch_size, " ")
    print('====> Epoch: {} Average loss: {:.4f} Accuracy: {:.4f} FeatLoss: {:.4f} '.format(
          epoch, train_loss / num_batches, train_acc/num_obs, feat_loss_total/num_batches))#len(train_loader.dataset)))
    #result_list.append([epoch,train_loss / batch_idx])

    
def test(epoch):
    model.eval()
    test_loss = 0
    test_acc = 0
    num_batch =0
    feat_loss_total = 0
    num_obs = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            init_observation, observation, final_observation, ref_init_state, ref_final_state, action = process_data(data)
            logpi, anchor_sentence = model(init_observation, observation, ref_init_state, ref_final_state)
            ctrl_loss = model.control_loss(logpi,action)
            positive_sentence = model.embed_task(init_observation, final_observation)
            embedding_loss = model.embedding_loss(anchor_sentence[1:], positive_sentence[1:], positive_sentence[:-1])
            loss = model.total_loss(embedding_loss, ctrl_loss)
            test_loss += loss.item()#/ len(image)
            acc = accuracy(logpi,action).item() #/len(image)
            test_acc += acc
            feat_loss_total += embedding_loss.item()
            num_obs += len(observation)
            num_batch +=1

    writer.add_scalar('Loss/test', test_loss/num_obs, epoch )
    writer.add_scalar('Regloss/test',feat_loss_total/num_obs, epoch)
    writer.add_scalar('Accuracy/test', test_acc/num_obs, epoch )
    print('====> Test set loss: {:.4f} Accuracy: {:.4f} FeatLoss: {:.4f} '.format(test_loss/num_batch, test_acc/num_obs, feat_loss_total/num_batch))#len(test_loader.dataset)))

def save_model(model, epoch, path):
    state_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    torch.save((state_dict, epoch, len(train_loader.dataset), opt_dict), path)


if __name__ == "__main__":
    filename = 'tecnet_4tasks'+'_images_feat'+str(args.feature_dim)+'_'+args.letter+'_{:06d}'.format(args.train_size)
    writer = SummaryWriter(comment=filename)
    basedirectory = 'octresults/'+filename
    print("Tensorboard logdir",  writer.logdir)
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