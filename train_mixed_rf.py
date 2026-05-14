import math
import os
import numpy as np
import argparse
from tensordict import TensorDict
from gvrp_env_rf import VRPEnvRF
from utils import set_seed, save_args_to_file
import torch
# torch.set_float32_matmul_precision('high')
import torch.optim as opt
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from data_augment import augment
import pickle


parser = argparse.ArgumentParser(description='start training')
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--epoch_size', type=int, required=True)
parser.add_argument('--nodes', type=int, required=True, help='N nodes')
parser.add_argument('--folder', type=str, required=True, help='folder to save logs & models')
parser.add_argument('--aug', type=int, default=1, help='aug during training')
parser.add_argument('--pomo', type=int, default=1, help='pomo number during training')
parser.add_argument('--multi_start', type=int, default=0, help='pomo multi-start training')
parser.add_argument('--batch', type=int)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--ft', type=str)
parser.add_argument('--rescale', type=int, default=0)
parser.add_argument('--div', type=int, default=1)
parser.add_argument('--resume', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--focus', nargs='+', type=str, default=[], help='List of focus strings')
parser.add_argument('--net', type=int, default=0)

reuse_encoder = None # 'logs/new_50-2-10/99.pt'

args = parser.parse_args()

print(args.focus)

folder = f'./logs/{args.folder}'

assert args.batch % args.div == 0

if args.net == 0:
    from routefinder_net_rf import Encoder, Decoder
else:
    from routefinder_net_rf_nolstm import Encoder, Decoder


def train_loop(rank, world_size):
    init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    assert args.div == world_size
    set_seed(args.seed + rank)

    device = f'cuda:{rank}'

    if rank == 0:
        sw = SummaryWriter(os.path.join('./logs', args.folder))

    env = VRPEnvRF(args.nodes, args.batch // args.div, device=device, aug=args.aug, n_samp=args.pomo)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    if args.ft:
        models = torch.load(args.ft, map_location='cpu')
        if isinstance(models, dict):
            encoder.load_state_dict(models['encoder'])
            decoder.load_state_dict(models['decoder'], strict=False)
        else:
            encoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[0].items()})
            decoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in models[1].items()})
        decoder.ft = True
        decoder.context_trans = nn.Sequential(
                nn.Linear(128, 8 * 128),
                nn.ReLU(),
                nn.Linear(8 * 128, 128)
            ).to(device)
        optimizer = opt.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)
    else:
        optimizer = opt.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    epoch_start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if isinstance(ckpt, dict):
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['decoder'])
            optimizer.load_state_dict(ckpt['optimizer'])
            epoch_start = ckpt['epoch'] + 1
        else:
            assert False
            encoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in ckpt[0].items()})
            decoder.load_state_dict({k.replace('module._orig_mod.', ''): v for k, v in ckpt[1].items()})

    # encoder = torch.compile(encoder)
    # decoder = torch.compile(decoder)
    encoder = DistributedDataParallel(encoder, device_ids=[rank])
    decoder = DistributedDataParallel(decoder, device_ids=[rank])

    # scaler = GradScaler(device='cuda')

    for epoch in range(epoch_start, args.epochs):
        # one epoch
        # assert args.epoch_size % args.batch == 0
        n_batchs = int(np.ceil(args.epoch_size / args.batch))

        tr = trange(n_batchs, desc=f'Epoch={epoch}') if rank == 0 else range(n_batchs)
        e_grad_norm_all = []
        d_grad_norm_all = []
        adv_logger = []
        # entropy_stat = np.zeros((21, ))
        if args.lr < 0:
            if epoch <= 2:
                lr = 1e-5
            # elif epoch >= 200:
            #     lr = (1 + math.cos((epoch - 200) / 100 * math.pi)) * 0.5 * 1e-4
            elif epoch >= 295:
                lr = 3e-6
            elif epoch >= 270:
                lr = 3e-5
            else:
                lr = 3e-4
        else:
            lr = args.lr
        # lr = 5e-5

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for this_it in tr:
            if args.focus:
                flags = args.focus[np.random.randint(0, len(args.focus))]
            else:
                flags = None
            td = env.reset(flags=flags)

            num_encoder_input = td['locs'].shape[0] // args.pomo
            node_embeddings, graph_embeddings = encoder(td[:num_encoder_input])
            node_embeddings = node_embeddings.repeat(args.pomo, 1, 1)
            graph_embeddings = graph_embeddings.repeat(args.pomo, 1)
            
            decoder.module.build_cache(node_embeddings)

            log_probs = []
            now_step = 0
            # first_acts = args.pomo != 1
            first_acts = args.multi_start != 0
            while not env.is_done():
                if first_acts:
                    acts = torch.arange(1, args.nodes + 1, device=device).long()
                    acts = acts.repeat_interleave(args.batch * args.aug // args.div) # TODO: only for POMO != 1 and AUG = 1
                    first_acts = False
                else:
                    logits = decoder(td, node_embeddings, graph_embeddings)
                    dist = torch.distributions.Categorical(logits=logits)
                    acts = dist.sample()
                    log_probs.append(dist.log_prob(acts))
                # entropy_stat[now_step] += dist.entropy().mean().item()
                now_step += 1
                td['action'] = acts
                td = env.step(td)
            rewards = env.get_reward()

            # rewards[rewards < -19] = rewards[rewards > -19].min()
            
            log_p = torch.stack(log_probs).transpose(0, 1)
            bl = rewards.view(-1, args.batch // args.div).mean(axis=0)
            bl = bl.repeat(args.aug * args.pomo)
            adv = rewards - bl

            loss = -log_p * adv.view(-1, 1)
            loss = loss.mean()
            loss.backward()

            e_grad_norm_before = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.)
            d_grad_norm_before = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.)

            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                with torch.no_grad():
                    adv_lv = adv.view(-1, args.batch // args.div)
                    adv_diff = adv_lv.max(axis=0).values - adv_lv.min(axis=0).values
                    adv_logger.append(adv_diff.mean().item())
                e_grad_norm_all.append(e_grad_norm_before.item())
                d_grad_norm_all.append(d_grad_norm_before.item())

        if rank == 0: # and (epoch % 10 == 9 or args.ft):
            torch.save({
                'epoch': epoch,
                'encoder': encoder.module.state_dict(),
                'decoder': decoder.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state()
            }, os.path.join(folder, f'{epoch}.pt'))

            
            sw.add_scalar('GradNorm/Encoder_Avg', np.mean(e_grad_norm_all), epoch)
            sw.add_scalar('GradNorm/Decoder_Avg', np.mean(d_grad_norm_all), epoch)
            sw.add_scalar('GradNorm/Encoder_Std', np.std(e_grad_norm_all), epoch)
            sw.add_scalar('GradNorm/Decoder_Std', np.std(d_grad_norm_all), epoch)
            sw.add_scalar('Adv/Adv_Avg', np.mean(adv_logger), epoch)
            sw.add_scalar('Adv/Adv_Std', np.std(adv_logger), epoch)
            sw.add_scalar('LR', lr, epoch)
    destroy_process_group()


def main():
    os.makedirs(os.path.join('./logs', args.folder), exist_ok=True)
    save_args_to_file(args, os.path.join('./logs', args.folder, 'cfg.json'))
    world_size = args.div
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 20000))
    mp.spawn(train_loop, (world_size, ), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()