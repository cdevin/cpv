import numpy as np
import pickle
import os
import cv2
import blosc
from gridworld.envs.grid_affordance import HammerWorld
from gridworld.policies.gridworld_policies import EatBreadPolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, PickupObjectPolicy, MoveObjectPolicy, GoToObjectPolicy, PickupSticksPolicy, GoToHousePolicy,  policy_dict, policy_inputs, policy_outputs, TASK_EVAL_MAP
from gridworld.policies.composite_policy import CompositePolicy, policies, policy_names, pol_index
from gridworld.policies.language import simple_language
import copy
import argparse
import random
import numpy as np
from gridworld.envs.grid_affordance import OBJECTS

ACTION_MAP = {'R': 0,
              'S': 1,
              'T': 2,
              'Q': 3,
              'p': 4,
              'd': 5,
              'e': 6,
              }
idx = 0
size = 10

RENDER_MODE = 'one_hot'
H2 = HammerWorld(res=3, visible_agent=True, use_exit=True, size=[size,size], render_mode=RENDER_MODE)
H= HammerWorld(add_objects =[],res=3, visible_agent=True, use_exit=True, agent_centric=False, goal_dim=0, size=[size,size], few_obj=False,
              render_mode=RENDER_MODE)
task_success = []

basedirectory = 'data/4tasks_onehot_pkl/' 

if not os.path.isdir(basedirectory):
    os.mkdir(basedirectory)
directory = basedirectory
policy_names = ['ChopTreePolicy', 'ChopRockPolicy', 'EatBreadPolicy', 'BuildHousePolicy', 'MakeBreadPolicy']
print(basedirectory)
saved = 0

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

episode = 0
list_of_samples = []

while saved < 100000:
    num_policies = random.randint(2,5)
    policy_list = np.random.choice(policy_names, size = num_policies)
    instruction = simple_language(policy_list)
    policy = CompositePolicy(policy_list, H.action_space, H2, noise_level=0.1)
    if episode % 1000 == 0 and len(task_success)> 0:
        with open(directory+'/episode{:04d}_'.format(episode)+'.pkl', 'wb') as f:
            pickle.dump(list_of_samples, f)
            saved += len(list_of_samples)
            list_of_samples = []
            
        print(episode, sum(task_success)/len(task_success), "saved", saved)
    if True:
        data= {}
        for style in ['exp']:
            step = 0
            d = False
            policy.reset()
            obs = H.reset(min_obj = policy.min_object_nums())
            H.episode = {'ref': 0, 'exp': 1}[style]
            agent = H.state['agent']
            init_state = copy.deepcopy(H.state)
            actions = []
            states = []
            dones = [False]
            a = policy.get_action(H.state)
            actions.append([a,agent[0], agent[1], False])
            images = [obs]
            step +=1
            while not d:
                obs, r,d,_ = H.step(int(a))
                agent = H.state['agent']
                dones.append(d)
                images.append(obs)
                if not d:
                    a = policy.get_action(H.state)
                else:
                    a = -1
                    
                if a is None:
                    a = 6
                actions.append([a,agent[0], agent[1], d])
                states.append(copy.deepcopy(H.state))
                step+=1
                
            final_state = copy.deepcopy(H.state)
            success = policy.eval_traj()
            task_success.append(success)
            ac_coef = 1+H.agent_centric
            if RENDER_MODE == 'rgb':
                img_size= (H.res*(H.nrow+1)*ac_coef, H.res*(H.ncol)*ac_coef, 3)
            elif RENDER_MODE == 'one_hot':
                img_size= (H.res*(H.nrow+1)*ac_coef, H.res*(H.ncol)*ac_coef, len(H.SPRITES))
            image_arr = []
            object_counts = {obj: [s['object_counts'][obj] for s in states] for obj in OBJECTS}
            counts_diff = get_counts_diff(object_counts)
            for i in range(len(images)):
                obs = np.reshape(images[i], img_size)
                w,h,c = obs.shape
                if actions[i][0] is None:
                    import pdb; pdb.set_trace()
                #obs[w-1, h-1, c-1] = int(actions[i][0])
                image_arr.append([obs])
            image_arr = np.concatenate(image_arr)
            actions = [ac[0] for ac in actions]
            #data[style] = (success, image_arr, counts_diff)
            if task_success:
                sample = (instruction, blosc.pack_array(image_arr), None, actions)
                list_of_samples.append(sample)
               # np.save(directory+'/episode{:04d}_'.format(episode)+style+'.npy', image_arr)
        episode += 1
print( "success rate", sum(task_success)/len(task_success), saved/episode)
