import numpy as np
import pickle
import os
import cv2
from gridworld.envs.grid_affordance import HammerWorld
from gridworld.policies.gridworld_policies import EatBreadPolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, PickupObjectPolicy, MoveObjectPolicy, GoToObjectPolicy, PickupSticksPolicy, GoToHousePolicy,  policy_dict, policy_inputs, policy_outputs, TASK_EVAL_MAP
from gridworld.policies.composite_policy import CompositePolicy, policies, policy_names, pol_index
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

H2 = HammerWorld(res=3, visible_agent=True, use_exit=True, size=[size,size])
H= HammerWorld(add_objects =[],res=3, visible_agent=True, use_exit=True, agent_centric=False, goal_dim=0, size=[size,size], few_obj=False)
task_success = []

for n in [1,2,4,8,16]:
    basedirectory = 'data/Oct_references_len'+str(n)+'/' 
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
    while saved < 20*n:
        num_policies = n 
        policy_list = np.random.choice(policy_names, size = num_policies)
        truncated_policy_list = [n[:-6] for n in policy_list]
        policy_n = '_'.join(truncated_policy_list)
        policy = CompositePolicy(policy_list, H.action_space, H2, noise_level=0.1)
        if episode % 10 == 0 and len(task_success)> 0:
            print(episode, sum(task_success)/len(task_success), "saved", saved)
        if True:
            data= {}
            for style in [ 'ref']:
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
                img_size= (H.res*(H.nrow+1)*ac_coef, H.res*(H.ncol)*ac_coef, 3)
                image_arr = []
                object_counts = {obj: [s['object_counts'][obj] for s in states] for obj in OBJECTS}
                counts_diff = get_counts_diff(object_counts)
                for i in range(len(images)):
                    obs = np.reshape(images[i], img_size)
                    w,h,c = obs.shape
                    if actions[i][0] is None:
                        import pdb; pdb.set_trace()
                    obs[w-1, h-1, c-1] = int(actions[i][0])
                    image_arr.append([obs])
                image_arr = np.concatenate(image_arr)
                actions = [ac[0] for ac in actions]
                data[style] = (success, image_arr, counts_diff)
            if data['ref'][0]:
                for style in ['ref']:
                    image_arr = data[style][1]
                    saved+=1
                    np.save(directory+'/'+policy_n+'_episode{:04d}_'.format(episode)+style+'.npy', image_arr)
                    object_counts = {obj: [s['object_counts'][obj] for s in states] for obj in OBJECTS}
                    with open(directory+'/'+policy_n+'_episode{:04d}_'.format(episode)+style+'_counts.pkl', 'wb') as f:
                        pickle.dump(object_counts, f)
            episode += 1
    print( "success rate", sum(task_success)/len(task_success), saved/episode)
