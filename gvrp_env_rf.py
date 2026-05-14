from line_profiler import profile

from generate_data_rf import generate_mtvrp_data, VARIANT_FEATURES
from data_augment import augment
import numpy as np
import torch
from tensordict import TensorDict
import sys
sys.path.append('cvrp_split_solver/build')
from gmsvrprf import vrp_split

import cProfile


def random_sample_variant():
    selects = [
        "CVRP", "OVRP", "VRPB", "VRPL", 
        "VRPTW", "OVRPTW", "OVRPB", "OVRPL", 
        "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", 
        "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
    ]
    index = np.random.randint(len(selects))
    return selects[index]


class VRPEnvRF(object):

    
    def __init__(self, n_nodes, batch_size, device='cpu', aug=1, n_samp=1):
        
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.device = device
        self.aug = aug
        self.samp = n_samp
        
        # real batch_size = self.batch_size * self.aug * self.samp

        self.locs = None
        self.demands = None
        self.tw = None
        self.distance_limit = None

        self.first_node = None
        self.current_node = None
        self.action_masks = None

        self.nd0 = None
        self.nd1 = None

        # O, B, L, TW
        self.flags = [False, False, False, False]

        self.actions = []

        self.batch_axis = torch.arange(self.batch_size * self.aug * self.samp, device=self.device)

        self.now_step = 0

    def reset(self, td=None, flags=None):
        self.now_step = 0

        self.actions.clear()
        if td is not None:
            data = td
            this_variant = flags
        else:
            if flags is None:
                this_variant = random_sample_variant()
            else:
                this_variant = flags
            data = generate_mtvrp_data(
                dataset_size=self.batch_size, 
                num_loc=self.n_nodes,
                variant=this_variant
            )

        locs = torch.from_numpy(data['locs'])
        locs_aug = augment(locs, self.aug)
        demand_linehaul = torch.from_numpy(data['demand_linehaul'])

        if 'demand_backhaul' in data:
            demand_backhaul = torch.from_numpy(data['demand_backhaul'])
            demands = demand_linehaul - demand_backhaul
        else:
            demands = demand_linehaul
        demands = torch.cat(
            [torch.zeros((demands.shape[0], 1), dtype=torch.float32), demands], axis=-1
        )[..., None]
        demands_aug = demands.repeat(self.aug, 1, 1)
        
        if 'time_windows' in data:
            tw = torch.from_numpy(data['time_windows'])
            ser = torch.from_numpy(data['service_time'])[..., torch.newaxis]
            tw = torch.cat([tw, ser], dim=-1)
            tw_aug = tw.repeat(self.aug, 1, 1)
        else:
            tw_aug = torch.zeros((self.batch_size * self.aug, self.n_nodes + 1, 3), dtype=torch.float32)
            tw_aug[:, :, 0] = 0.0   # start time limit 0
            tw_aug[:, :, 1] = -1.0  # no end time limit
            tw_aug[:, :, 2] = 0.0   # service time 0

        if 'distance_limit' in data:
            distance_limit = data['distance_limit']
            distance_limit_aug = torch.from_numpy(distance_limit)[:, None, :].repeat(self.aug, self.n_nodes + 1, 1)
        else:
            distance_limit_aug = torch.ones((self.batch_size * self.aug, self.n_nodes + 1, 1), dtype=torch.float32) * -1  # no limit

        self.flags = [VARIANT_FEATURES[this_variant][key] for key in ["O", "B", "L", "TW"]]
        
        locs_samp = locs_aug.repeat(self.samp, 1, 1)
        demands_samp = demands_aug.repeat(self.samp, 1, 1)
        tw_samp = tw_aug.repeat(self.samp, 1, 1)
        distance_samp = distance_limit_aug.repeat(self.samp, 1, 1)
        
        self.locs = locs_samp.to(self.device).float()
        self.demands = demands_samp.to(self.device).float()
        self.tw = tw_samp.to(self.device).float()
        self.distance_limit = distance_samp.to(self.device).float()

        self.first_node = torch.zeros((self.batch_size * self.aug * self.samp, ), dtype=torch.int64).to(self.device)
        self.current_node = torch.zeros((self.batch_size * self.aug * self.samp, ), dtype=torch.int64).to(self.device)
        self.action_masks = torch.zeros((self.batch_size * self.aug * self.samp, self.n_nodes + 1), dtype=torch.bool).to(self.device)
        self.action_masks[:, 0] = 1  # cannot visit depot

        self.nd0 = torch.zeros(self.batch_size * self.aug * self.samp, dtype=torch.float32).to(self.device)
        self.nd1 = torch.zeros(self.batch_size * self.aug * self.samp, dtype=torch.float32).to(self.device)

        td = TensorDict({
            'locs': self.locs,
            'demand': self.demands,
            'time_windows': self.tw,
            'distance_limit': self.distance_limit,
            'first_node': self.first_node,
            'current_node': self.current_node,
            'mask': self.action_masks,
            'flags': torch.from_numpy(np.array(self.flags, dtype=np.float32)).to(self.device).repeat(self.batch_size * self.aug * self.samp, 1),

            'nd0': self.nd0,
            'nd1': self.nd1
        }, batch_size=(self.batch_size * self.aug * self.samp, ), device=self.device)
        return td
    
    def step(self, td):
        self.now_step += 1
        self.actions.append(td['action'].clone())
        
        action = td['action'].clone()
        self.action_masks.scatter_(1, action.unsqueeze(1), 1)
        demands = self.demands[self.batch_axis, action].squeeze()

        pos_mask = demands > 0
        self.nd0 += pos_mask * demands
        neg_mask = demands < 0
        self.nd1 -= neg_mask * demands
        # self.nd0[demands > 0] += demands[demands > 0]
        # self.nd1[demands < 0] -= demands[demands < 0]

        td['current_node'] = action
        td['mask'] = self.action_masks
        td['nd0'] = self.nd0
        td['nd1'] = self.nd1
        return td

    def is_done(self):
        return self.now_step == self.n_nodes
        print(self.now_step, self.n_nodes)
        return torch.all(self.action_masks == 1).item()
    
    def get_reward(self):
        acts_np = torch.stack(self.actions).transpose(0, 1).contiguous().cpu().numpy().astype(np.int32)
        locs_np = self.locs.cpu().numpy().astype(np.float32)
        demands_np = self.demands.cpu().numpy().astype(np.float32)
        tw_np = self.tw.cpu().numpy().astype(np.float32)
        distance_limit = self.distance_limit.cpu().numpy().astype(np.float32)

        data = np.concat([locs_np, demands_np, tw_np, distance_limit], axis=-1).astype(np.float32)
        split_res = vrp_split(data, acts_np, self.flags)
        split_res[split_res >= 999] = -500
        split_res = split_res.reshape(self.aug * self.samp, self.batch_size)
        instance_max = split_res.max(axis=0, keepdims=True).repeat(self.aug * self.samp, axis=0)
        instance_max[instance_max <= -500] = 500
        mask = split_res <= -500
        split_res[mask] = instance_max[mask]
        split_res = split_res.reshape(-1)

        # if mask.sum() != 0:
        #     print('Note: refine rewards!')
        return -torch.from_numpy(split_res).to(self.device)


def main():
    from routefinder_net_rf import Encoder, Decoder

    device = 'cuda:0'
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    p = cProfile.Profile()

    p.enable()
    with torch.inference_mode():
        env = VRPEnvRF(n_nodes=50, batch_size=100, device=device, aug=8, n_samp=50)
        for _ in range(10):
            td = env.reset(td=None)
            
            batch_size = td['locs'].shape[0]
            encoder_input_size = batch_size // env.samp
            node_embeddings, graph_embeddings = encoder(td[:encoder_input_size])
            node_embeddings = node_embeddings.repeat(env.samp, 1, 1)
            graph_embeddings = graph_embeddings.repeat(env.samp, 1)
            decoder.build_cache(node_embeddings)

            for i in range(50):
                # td['action'] =  i * torch.ones((env.batch_size * env.aug * env.samp,), dtype=torch.int64) + 1
                logits = decoder(td, node_embeddings, graph_embeddings)
                probs = torch.softmax(logits, dim=-1)
                acts = torch.multinomial(probs, 1).squeeze(-1)
                td['action'] = acts
                td = env.step(td)
            reward = env.get_reward()
            # print("Reward:", reward)
    p.disable()
    p.dump_stats('env_test.prof')


if __name__ == "__main__":
    main()
