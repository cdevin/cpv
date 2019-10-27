from gridworld.policies.gridworld_policies import EatBreadPolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, PickupObjectPolicy, MoveObjectPolicy, GoToObjectPolicy, PickupAxePolicy, PickupHammerPolicy, PickupSticksPolicy, GoToHousePolicy,MakeBreadPolicy,  policies, policy_names, policy_dict, policy_inputs, policy_outputs, TASK_EVAL_MAP, policy_removes
from gridworld.envs.grid_affordance import OBJECTS
import copy
#OBJECTS =['rock', 'hammer', 'tree', 'axe', 'bread', 'sticks', 'house']
import copy
policies = [EatBreadPolicy, GoToHousePolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, PickupSticksPolicy, PickupAxePolicy, PickupHammerPolicy, MakeBreadPolicy]
policy_names = ['EatBreadPolicy', 'GoToHousePolicy', 'ChopTreePolicy', 'BuildHousePolicy', 'ChopRockPolicy', 'PickupSticksPolicy','PickupAxePolicy', 'PickupHammerPolicy', 'MakeBreadPolicy']
policy_dict = {k:v for k,v in zip(policy_names, policies)}
pol_index = {'ChopTreePolicy': 2, 'BuildHousePolicy': 3, 'EatBreadPolicy': 0, 
               'PickupSticksPolicy': 5, 'ChopRockPolicy': 4, 'PickupHammerPolicy': 1,
               'PickupAxePolicy': 7, 'MakeBreadPolicy': 8,
              }
policy_names = list(pol_index.keys())
class CompositePolicy():
    def __init__(self, policy_list, action_space, env, noise_level=0.0):
        self.policy_list = policy_list
        self.action_space = action_space
        self.env = env
        self.policies = [policy_dict[policy_n](action_space, env, noise_level=noise_level) for policy_n in policy_list]
        self.policy_index = 0
        self.eval = 0
        self.mid_states = []
        self.step = 0
        self.noise_level = noise_level
        
    def get_action(self, state):
        #print("policy", self.policy_list[self.policy_index], self.policies[self.policy_index])
        if len(self.mid_states) == 0:
            self.mid_states.append(copy.deepcopy(state))
        a = self.policies[self.policy_index].get_action(state)
        self.step+=1
        #print(self.policy_list[self.policy_index], a)
        #print("actions", a)
        #print("polucy",self.policy_list[self.policy_index] )
        while (a == None or a == 6) and self.policy_index < (len(self.policies)-1):
            eval = TASK_EVAL_MAP[self.policy_list[self.policy_index]](self.mid_states[-1], state) == 1
            #print("prev state", self.mid_states[-1], len(self.mid_states))
            #print("curr state", state)
            #print("policy", self.policy_list[self.policy_index], "step", self.step, "success", eval)
            
            self.mid_states.append(copy.deepcopy(state))
            self.eval += eval
            self.policy_index +=1
            a = self.policies[self.policy_index].get_action(state)
            
        if (a == None or a == 6) and self.policy_index == (len(self.policies)-1):
            eval = TASK_EVAL_MAP[self.policy_list[self.policy_index]](self.mid_states[-1], state) == 1
            self.eval += eval
            #print("policy", self.policy_list[self.policy_index], "step", self.step, "success", eval)
        if a == None:
            a = 6
        return a
    
    def reset(self):
        [pol.reset() for pol in self.policies]
        self.policy_index = 0
        self.eval = 0
        self.mid_states = []
        
    def min_object_nums(self):
        objects = {o:0 for o in OBJECTS}
        needed_objects = {o:0 for o in OBJECTS}
        for pol in self.policy_list:
            for obj in policy_inputs[pol]:
                if objects[obj] == 0:
                    needed_objects[obj] += 1
                    objects[obj] += 1
            for obj in policy_outputs[pol]:
                objects[obj] += 1
            for obj in policy_removes[pol]:
                objects[obj] -= 1
                if objects[obj] < 0:
                    import pdb; pdb.set_trace()
        return needed_objects
    
    def eval_traj(self):
        #import pdb; pdb.set_trace()
        return self.eval == len(self.policy_list)
        