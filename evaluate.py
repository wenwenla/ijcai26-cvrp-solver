# %%
import argparse
from collections import defaultdict
from tensordict import TensorDict
import time
from tqdm import tqdm, trange

from gvrp_env_rf import VRPEnvRF
import pickle
import torch
import numpy as np

from utils import rollout_with_agents
from data_augment import augment, augment_xy_data_by_N_fold
from utils import set_seed
import os


torch.set_float32_matmul_precision('high')

# %%
device = 'cuda:0'

n_nodes = 50
n_aug = 8
total_instance = 1000
test_batch_size = 100

model_ckpt = 'logs/debug-50/299.pt'
# O B L TW
flags = 'CVRP'
temperature = 1.6
ft = False

net = 0

if net == 0:
    from routefinder_net_rf import Encoder, Decoder
else:
    from routefinder_net_rf_nolstm import Encoder, Decoder

# %%
def get_dist(locs, acts):
    prev = locs[0]
    dist = 0
    for a in acts:
        dist += np.linalg.norm(locs[a] - prev, ord=2)
        prev = locs[a]
    dist += np.linalg.norm(prev - locs[0], ord=2)
    return dist.item()


# %%
variants = [
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", 
    "OVRPB", "OVRPL", "OVRPTW",
    "VRPBL", "VRPBTW", "VRPLTW", 
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW",
    "OVRPBLTW"
]

variants = ['CVRP']
encoder = Encoder().to(device)
decoder = Decoder().to(device)

models = torch.load(model_ckpt, map_location='cpu')
encoder.load_state_dict(models['encoder'])
decoder.load_state_dict(models['decoder'])

for f in variants:
    set_seed(19971023)
    start = time.time()

    env = VRPEnvRF(n_nodes, test_batch_size, device, aug=n_aug, n_samp=n_nodes)
    flags = f
    testdata = f'data/{f.lower()}/test/{n_nodes}.npz'
    td = np.load(testdata)

    all_rews = []

    
    # with torch.profiler.profile(
    # activities=[torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as prof:
    r_tot = 0
    for b_idx in range(total_instance // test_batch_size):
        this_td = {}
        for key in td.keys():
            this_td[key] = td[key][b_idx * test_batch_size:b_idx * test_batch_size + test_batch_size]

        # first_acts = torch.arange(0, n_nodes, device=device) + 1
        # first_acts = first_acts.repeat_interleave(test_batch_size).repeat(n_aug)
        first_acts = None

        rew, act, r_t = rollout_with_agents(env, encoder, decoder, this_td, 'sampling', fixed_start=first_acts, temperature=temperature, flags=flags)
        max_rew, max_idx = rew.view(-1, test_batch_size).max(dim=0)
        all_rews.append(max_rew.cpu().tolist())
        r_tot += r_t

    print(f'{flags}: {-np.mean(all_rews):.3f}')
    # prof.export_chrome_trace('rollout_trace_final.json')
    end = time.time()
    # print(f'Time taken: {end - start:.2f} seconds, R/T: {r_tot / (end - start):.4f} seconds')
