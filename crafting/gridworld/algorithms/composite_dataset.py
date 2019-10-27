import torch
import glob
#from skimage import io, transform
import cv2
from torchvision import transforms, utils
import numpy as np
import pickle
from natsort import natsorted
MAX_ACTIONS = 7
import time
class ActionToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def convert_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.permute(2,0,1)
            return image.type(torch.FloatTensor)/255.
        else: 
            image[-1,-1,-1] = 0 
            image = image.transpose((2, 0, 1))
            return torch.from_numpy(image).type(torch.FloatTensor)/255.

    def __call__(self, sample):
        image = sample['image']#, sample['last_image']
        action = image[-1,-1,-1]
        for k,v in sample.items():
            if 'image' in k:
                sample[k] = self.convert_image(v)
        
        if 'ref_middle' in sample:
            sample['ref_middle'] = torch.stack([self.convert_image(i) for i in sample['ref_middle']])
        if 'exp_middle' in sample:
            sample['exp_middle'] = torch.stack([self.convert_image(i) for i in sample['exp_middle']])
        #import pdb; pdb.set_trace()
        #limage = limage.transpose((2, 0, 1))
        #print({n: v.shape for n,v in sample.items()})
        sample['action'] = int(action)
        return sample

class CompositeDataset(torch.utils.data.Dataset):
    def __init__(self, directory='/persistent/affordance_world/data/paired_compositions2/',
                 train=True, size=None, include_ref=True, pickup_balance=0, is_labeled=False,
                 num_middle_states=0,
                ):
        self.directory = directory
        self.transform = ActionToTensor()
        self.include_ref = include_ref
        self.pickup_balance = pickup_balance
        self.is_labeled = is_labeled
        self.train = train
        self.num_middle_states =num_middle_states
        if train:
            traj_files = natsorted(glob.glob(directory+'/episode*[1-9]_*.npy'))
            print(len(traj_files))
        else:
            traj_files = natsorted(glob.glob(directory+'/episode*0_*.npy'))
            print(len(traj_files))            
        print("gathering paths", len(traj_files))
        if include_ref:
            coef = 2
        else: 
            coef = 1
        print("coef is ", coef)
        if size is None:
            size = int(len(traj_files)/coef)
            if size * coef != len(traj_files):
                print("unenven number of traj_files")
        if size*coef > len(traj_files):
            size = int(len(traj_files)/coef)
        self.reference_files = []#0 for i in range(size)]
        self.expert_files = []#0 for i in range(size)]
        print("size", size)
        for i in range(size*coef):
            file = traj_files[i]
            dirs = file.split('/')
            words = dirs[-1].split('_')
            episode = int(words[0][7:])
#             if episode != int(i/2)+1:# and episode != int(i/2) :
#                 print("episode", episode, "file", file, "len", len(self.reference_files))
#                 import pdb; pdb.set_trace()
#             if episode >= len(self.reference_files):
#                 print("episode", episode, "file", file, "len", len(self.reference_files))
#                 import pdb; pdb.set_trace()
            if 'ref' in file:
                self.reference_files.append((file, episode))
            elif 'exp' in file:
                self.expert_files.append((file, episode))
        if 0 in self.reference_files  or 0 in self.expert_files:
            print("0 in ref or exp files")
            import pdb; pdb.set_trace()
        if self.include_ref:
            for i in range(len(self.expert_files)):
                #print(i, self.expert_files[i], self.4reference_files[i])
                assert(self.expert_files[i][-1] == self.reference_files[i][-1])
        print("done gathering paths")
#         self.time_avg = 0
#         self.num_steps = 0
        
    def __len__(self):
        return len(self.expert_files)

    def _get_label(self, path):
        return int(path.split('_')[-2])
    
    def _get_middles(self, trajectory):
        trajectory = list(trajectory)
        x = int(len(trajectory)/self.num_middle_states)
        if x == 0:
            middle = trajectory
        else:
            middle = trajectory[::x][1:]
        while len(middle) < self.num_middle_states:
            middle = [trajectory[0]]+middle
        while len(middle) > self.num_middle_states:
            middle = middle[:-1]
        #print(len(middle))
        #print([t.shape for t in middle])
        return middle
        
    def __getitem__(self, idx):
        #t1 = time.clock()
        
        ref_is_exp = np.random.randint(0,2)
        
        if ref_is_exp:
            ref_files = self.expert_files
            exp_files = self.reference_files
        else:
            exp_files = self.expert_files
            ref_files = self.reference_files
        exp = np.load(exp_files[idx][0])

            
        #if len(exp) < 3:
        #    print("idx", idx, "exp length is", exp.shape, "file", self.expert_files[idx][0])
        if len(exp) < 4:
            exp = [t for t in exp]
            while len(exp) < 4:
                exp = [exp[0]] + exp
        index = np.random.randint(0, len(exp)-2)
        if self.include_ref:
            ref = np.load(ref_files[idx][0])
            
            ref_init = ref[0]
            ref_final = ref[-1]
            sample = {'image': exp[index], 
                      'init_image': exp[0],
                      'final_image': exp[-1],
                      'post_image': ref_final , 
                      'pre_image': ref_init, 
            }
            if self.num_middle_states >0:
                sample['ref_middle'] = self._get_middles(ref)
                sample['exp_middle'] = self._get_middles(exp)
            
        else:
            sample = {'image': exp[index], 
                      'init_image': exp[0],
                      'final_image': exp[-1],
                     }
        if self.is_labeled:
            sample['task'] = self._get_label(exp_files[idx][0])
            
        sample = self.transform(sample)

        if sample['action'] == 6:
            print("idx", idx, "action", sample['action'], "exp length is", len(exp), "file", exp_files[idx][0])
            # Hack to avoid crashing, need to remove these tiny samples
            sample['action'] = 5
        return sample


class StateActionToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def convert_image(self, image):
        if image.shape[0] == 33:
            image = image[:32]
        if isinstance(image, torch.Tensor):
            #image = image.permute(2,0,1)
            return image.type(torch.FloatTensor)/10 -0.5
        else:
            return torch.from_numpy(image).type(torch.FloatTensor)/10 -0.5

    def __call__(self, sample):
        image = sample['image']#, sample['last_image']
        action = image[-1]
        for k,v in sample.items():
            if 'image' in k:
                sample[k] = self.convert_image(v)
                #print(sample[k])
#         if 'ref_middle' in sample:
#             sample['ref_middle'] = torch.stack([self.convert_image(i) for i in sample['ref_middle']])
#         if 'exp_middle' in sample:
#             sample['exp_middle'] = torch.stack([self.convert_image(i) for i in sample['exp_middle']])
        #import pdb; pdb.set_trace()
        #limage = limage.transpose((2, 0, 1))
        #print({n: v.shape for n,v in sample.items()})
        sample['action'] = int(action)
        return sample
    
class StateCompositeDataset(CompositeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform = StateActionToTensor()
    
class IRLDataset(CompositeDataset):
    def  __getitem__(self, idx):
        idx = idx % len(self.expert_files)
        exp = np.load(self.expert_files[idx][0])

        if len(exp) < 3:
            print("idx", idx, "exp length is", exp.shape, "file", self.expert_files[idx][0])
        if len(exp) < 4:
            exp = [t for t in exp]
            while len(exp) < 3:
                exp = [exp[0]] + exp
        index = np.random.randint(0, len(exp)-2)
        sample ={'image': exp[index], 
                 'next_image': exp[index+1]}
        sample = self.transform(sample)
        sample['action'] = np.eye(7)[sample['action']].astype(np.float32)
        return sample
    def __len__(self):
        return len(self.expert_files)*15

