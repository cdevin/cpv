import numpy as np
import pickle
import torch
import os
import cv2
import time
import torch
from gridworld.envs.grid_affordance import HammerWorld, OBJECTS
from gridworld.policies.gridworld_policies import EatBreadPolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, PickupObjectPolicy, MoveObjectPolicy, GoToObjectPolicy, PickupAxePolicy, PickupHammerPolicy, GoToHousePolicy, policies, policy_names, policy_dict, policy_inputs, policy_outputs, TASK_EVAL_MAP
from gridworld.policies.composite_delta_policy import LearnedBCPolicy
from gridworld.envs.grid_affordance import HammerWorld
from gridworld.policies.composite_policy import CompositePolicy, policies, policy_names, pol_index

from gridworld.algorithms.composite_models import CompositeDotModelV3, NaiveModel, TaskEmbeddingModel
from gridworld.algorithms.task_embeddings_networks import ImageTaskEmbeddingModel

from gridworld.policies.task_embeddings_policy import LearnedTECNetPolicy
from gridworld.policies.task_embeddings_policy import LearnedTECNetPolicy
import random
import copy
import glob
import argparse
#H= HammerWorld()
import glob
import pickle
import natsort
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--task', type=str,
#                     help='task name', default=None)
parser.add_argument('--tasks', type=str, nargs="*", default=None,
                    help='which task to run')
parser.add_argument('--model', type=str,
                    help='exp name')
parser.add_argument('--tb', type=str, default=None,
                    help='tensorboard dir')
parser.add_argument('-N', type=int, default=1,
                    help='number of rollouts')
parser.add_argument('--render', type=str,default=False,
                    help='task name')
parser.add_argument('--type', type=str, choices=['V3', 'TEC', 'Naive', "TE", "Duan"])
parser.add_argument('--start-idx', type=int, default=0,
                    help='model index to start at')
parser.add_argument('--end-idx', type=int, default=None,
                    help='model index to end at')
parser.add_argument('--num-policies', type=int, default=4,
                    help='length of trajecotries to eval on')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu')
parser.add_argument('--num-sum',  type=int, default=1,
                    help='sumembeddings')
parser.add_argument('--feature-dim', type=int, default=256)
parser.add_argument('--avoidant',action='store_true', default=False,
                    help='use avoidant dataset')
parser.add_argument('--relu', action='store_true', default=False,
                    help='relu the representation')
args = parser.parse_args()
RENDER = args.render
print(args)
policytype,modeltype={'V3': (LearnedBCPolicy, CompositeDotModelV3),
                      'TEC': (LearnedTECNetPolicy,ImageTaskEmbeddingModel),
                      'Naive': (LearnedBCPolicy, NaiveModel),
                      'TE': (LearnedTECNetPolicy, TaskEmbeddingModel)
                     }[args.type]

print(policytype, modeltype)

ACTION_MAP = {'R': 0,
              'S': 1,
              'T': 2,
              'Q': 3,
              'p': 4,
              'd': 5,
              'e': 6,
              }
idx = 0
max_path_lenths = {1:90, 2: 90, 4:160, 8:280, 16:550}

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
if args.tb is not None:
    writer = SummaryWriter(logdir=args.tb)
else: 
    writer = SummaryWriter(logdir='evals/'+args.model)

policy_task_list = ['EatBreadPolicy',
                   'ChopTreePolicy',
                   'BuildHousePolicy',
                   'MakeBreadPolicy',
                   'ChopRockPolicy',
                   ]
EMBEDDING_MODELS =  ['V3', 'TEC', 'TE']
CONCAT_MODELS = [] 

def eval_counts(ref_counts_diff, counts_diff):
    counts_diff = copy.deepcopy(counts_diff)
    success =1
    for obj in ref_counts_diff.keys():
        for t, diff in enumerate(ref_counts_diff[obj]):
            if diff != 0:
                if diff in counts_diff[obj]:
                    i = counts_diff[obj].index(diff)
                    counts_diff[obj][counts_diff[obj].index(diff)] = 0
                else:
                    success = 0
    return success

def get_counts_diff(counts):
    return {obj: [si-sj for si,sj in zip(counts[obj][1:],counts[obj][:-1] )] for obj in counts.keys()}

task_id = pol_index
if args.tasks is not None:
    policy_task_list = args.tasks
print(policy_task_list)
H= HammerWorld(add_objects =[],res=3, visible_agent=True, use_exit=True, agent_centric=False, goal_dim=0, size=[10,10], pretty_renderable=True)

def process_path(policy, path, num_pol, num_sum):
    images = np.load(path)
    policy_list = path.split('/')[-1].split('_')[:num_pol]
    policy_list = [p+'Policy' for p in policy_list]
    policy_n = ''.join(policy_list)
    counts_path = path[:-4]+'_counts.pkl'
    with open(counts_path, 'rb') as f:
        ref_counts = pickle.load(f)
    ref_counts_diff = get_counts_diff(ref_counts)
    last_img =images[-1]
    first_img = images[0]
    if args.type in EMBEDDING_MODELS: 
        goal_feat = policy.get_goal_feat(first_img, last_img).detach()
        return policy_list, policy_n, ref_counts_diff, goal_feat
    elif args.type in CONCAT_MODELS:
        return policy_list, policy_n, ref_counts_diff, images
    else:
        return policy_list, policy_n, ref_counts_diff, first_img, last_img


def process_path_list(policy, path_list, num_pol, num_sum):
    #print("Using list system")
    ref_counts_diff_total = {}
    policy_list_total = []
    policy_n_total = ''
    goal_feat_total = None
    first_imgs = None
    last_imgs = None
    for path in path_list:
        if args.type in EMBEDDING_MODELS: 
            policy_list, policy_n, ref_counts_diff, goal_feat = process_path(policy, path, num_pol, num_sum)
            if goal_feat_total is None:
                goal_feat_total = goal_feat
            else: 
                goal_feat_total += goal_feat
        elif args.type in CONCAT_MODELS:
            policy_list, policy_n, ref_counts_diff, images = process_path(policy, path, num_pol, num_sum)
            if first_imgs is None:
                first_imgs = images
            else:
                first_imgs =  np.concatenate([first_imgs, images], axis=0)
        else:
            policy_list, policy_n, ref_counts_diff, first_img, last_img = process_path(policy, path, num_pol, num_sum)
            if first_imgs is None:
                first_imgs = first_img
                last_imgs = last_img
            else:
                first_imgs += first_img
                last_imgs += last_img
        
        for obj in ref_counts_diff.keys():
            if obj in ref_counts_diff_total:
                ref_counts_diff_total[obj] += ref_counts_diff[obj]
            else:
                ref_counts_diff_total[obj] = ref_counts_diff[obj]
        policy_list_total += policy_list
        policy_n_total += policy_n
    if args.type in EMBEDDING_MODELS: 
        if 'TEC' in args.type:
            goal_feat_total = goal_feat_total/torch.norm(goal_feat_total)
        return policy_list_total, policy_n_total, ref_counts_diff_total, goal_feat_total
    elif args.type in CONCAT_MODELS:
        return policy_list_total, policy_n_total, ref_counts_diff_total, first_imgs
    else:
        return policy_list_total, policy_n_total, ref_counts_diff_total, first_imgs, last_imgs/num_sum


def run_model(policy, path, model_epoch, num_pol, num_sum):
    if args.type in EMBEDDING_MODELS: 
        if isinstance(path, list):
             policy_list, policy_n, ref_counts_diff, goal_feat = process_path_list(policy, path, num_pol, num_sum)
        else:
            policy_list, policy_n, ref_counts_diff, goal_feat = process_path(policy, path, num_pol, num_sum)
    elif args.type in CONCAT_MODELS:
        if isinstance(path, list):
             policy_list, policy_n, ref_counts_diff, first_img = process_path_list(policy, path, num_pol, num_sum) # first_img is actually list of all images
        else:
            policy_list, policy_n, ref_counts_diff, first_img = process_path(policy, path, num_pol, num_sum) # first_img is actually list of all images
    else:
        if isinstance(path, list):
            policy_list, policy_n, ref_counts_diff, first_img, last_img = process_path_list(policy, path, num_pol, num_sum)
        else:
            policy_list, policy_n, ref_counts_diff, first_img, last_img = process_path(policy, path, num_pol, num_sum)
    composite_policy = CompositePolicy(policy_list, H.action_space, H, noise_level=0.)
    task_success = []
    num_steps = []
    diffs = []
    max_path_lenth = max_path_lenths[num_pol*num_sum]
    for episode in range(0, args.N):
        if True:
            step = 0
            d = False
            H.episode = episode
            obs = H.reset(min_obj = composite_policy.min_object_nums())
            init_obs = obs.reshape(33,30,3)
            agent = H.state['agent']
            init_state = copy.deepcopy(H.state)
            images = [obs]
            if RENDER:
                H.pretty_render()
                H.render()
            actions = []
            
            dones = [False]
            init_state = copy.deepcopy(H.state)
            states = [init_state]
            if args.type in EMBEDDING_MODELS: 
                a = policy.get_action_from_ref(obs, init_obs, goal_feat)
            elif args.type in CONCAT_MODELS:
                a = policy.get_action(images, first_img)
            else:
                a = policy.get_action(obs, first_img, last_img, init_obs)

            actions.append([a,agent[0], agent[1], False])
            #if max(obs.shape) > 100:
            #    import pdb; pdb.set_trace()
            step +=1
            while not d and step < max_path_lenth:
                if a is None:
                    a = 6
                obs, r,d,_ = H.step(int(a))
                if RENDER:
                    H.pretty_render()
                    H.render()
                agent = H.state['agent']
                dones.append(d)
                images.append(obs)
                states.append(copy.deepcopy(H.state))
                if not d:
                    if args.type in EMBEDDING_MODELS: 
                        a = policy.get_action_from_ref(obs, init_obs, goal_feat)
                    elif args.type in CONCAT_MODELS:
                        a = policy.get_action(images, first_img)
                    else:
                        a = policy.get_action(obs, first_img, last_img, init_obs)
                else:
                    a = -1

                actions.append([a,agent[0], agent[1], d])
                final_state = copy.deepcopy(H.state)
                step+=1
                
                object_counts = {obj: [s['object_counts'][obj] for s in states] for obj in OBJECTS}
                counts_diff = get_counts_diff(object_counts)
                success = eval_counts(ref_counts_diff, counts_diff)
                if success:
                    d = True
            object_counts = {obj: [s['object_counts'][obj] for s in states] for obj in OBJECTS}
            counts_diff = get_counts_diff(object_counts)
            success = eval_counts(ref_counts_diff, counts_diff)
            task_success.append(success)
            if success:
                num_steps.append(step)
    return sum(task_success)/len(task_success), num_steps

model_paths = natsort.natsorted(glob.glob(args.model+'*0.pt'))

def get_epoch(path):
    epoch = []
    for c in path[::-1]:
        if c.isdigit():
            epoch.append(c)
        else:
            if len(epoch) > 0:
                return int(''.join(epoch[::-1]))

print("found", len(model_paths), "models")

max_epoch = args.start_idx-1
while True:
    model_paths = natsort.natsorted(glob.glob(args.model+'*0.pt'))
    for path in model_paths:
        epoch = get_epoch(path)
        if epoch > max_epoch:
            num_sleeps = 0
            max_epoch = epoch
            print(epoch)
            if epoch >= args.start_idx and epoch %40 == 0:

                print(path, epoch)
                if args.cpu :
                    state_dict, epoch, num_per_epoch , _= torch.load(path,  map_location='cpu')
                else:
                    state_dict, epoch, num_per_epoch , _= torch.load(path)
                policy = policytype(state_dict, device, model =modeltype , agent_centric=False, env_dim =(H.res*(H.nrow+1), H.res*(H.ncol), 3), task_embedding_dim=args.feature_dim, relu=args.relu)
                for num_pol, num_sum in [(1,1), (1,2),(2,1), (2,2),(2,4),(4,1), (4,2),(8,1), (16,1)]:
                    successes = []
                    num_steps = []
                    example_references= 'data/Oct_references_len'+str(num_pol)+'/'
                    task_paths = glob.glob(example_references  +'*.npy')
                    if len(task_paths) == 0:
                        print("gllob was empty", example_references +'*')
                    print("num_pol", num_pol, "num_sum", num_sum)
                    for task_path in task_paths:
                        if num_sum > 1:
                            sampled_task_paths = random.sample(task_paths, k = num_sum)
                            success, steps = run_model(policy,sampled_task_paths, epoch, num_pol, num_sum)
                        else:
                            success, steps = run_model(policy,task_path, epoch, num_pol, num_sum)
                        successes.append(success)
                        if len(steps) > 0:
                            num_steps.append(sum(steps)/len(steps))
                    if len(num_steps) > 0:
                        avg_length =sum(num_steps)/len(num_steps)
                    else:
                        avg_length = 0
                    writer.add_scalar("onpolicy/{}pol_{}sum".format(num_pol, num_sum),sum(successes)/len(successes), epoch)
                    print("epoch", epoch, "task","Average", "success rate", sum(successes)/len(successes), "average length", avg_length)
    print('waiting')
    time.sleep(60)
    num_sleeps += 1
    if num_sleeps > 30:
        break

