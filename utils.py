import sys
import time

sys.path.append('cvrp_split_solver/build')
import json
import torch
from tensordict import TensorDict
import numpy as np
import random
from torch.amp import autocast


def set_seed(seed):
    """
    Set the random seed for reproducibility in PyTorch and other libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch on CPU
    torch.cuda.manual_seed(seed)  # For PyTorch on the current GPU
    torch.cuda.manual_seed_all(seed)  # For PyTorch on all GPUs
    # torch.backends.cudnn.deterministic = True  # Ensure deterministic results with cuDNN
    # torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility


def save_args_to_file(args, file_path):
    """
    保存 argparse 的参数到文件。
    :param args: argparse.Namespace 对象
    :param file_path: 保存文件的路径
    """
    args_dict = vars(args)  # 转换为字典
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)  # 保存为 JSON 格式，带缩进
    print(f"Arguments saved to {file_path}")


# def select_node_embedding(node_embeddings, index):
#     '''
#     node_embeddings: tensor (batch_size, n_nodes, embedding_size)
#     index: tensor (batch_size, )

#     ret: tensor (batch_size, embedding_size)
#     '''
#     batch_size = node_embeddings.shape[0]
#     return node_embeddings[torch.arange(batch_size), index, :]


def select_node_embedding(node_embeddings, index):
    """
    node_embeddings: tensor (batch_size, n_nodes, embedding_size)
    index: tensor (batch_size, )

    return: tensor (batch_size, embedding_size)
    """
    batch_size, n_nodes, emb_size = node_embeddings.shape

    # reshape index 为 (batch_size, 1, 1)
    index = index.view(batch_size, 1, 1).expand(-1, 1, emb_size)

    # 使用 gather 取出每个 batch 对应节点
    selected = torch.gather(node_embeddings, dim=1, index=index)

    # squeeze 去掉节点维度
    return selected.squeeze(1)


def rollout_with_agents(env, encoder, decoder, td, rollout_type, temperature=1, fixed_start=None, flags=None):

    td = env.reset(td, flags)
    actions = []
    encoder.eval()
    decoder.eval()
    first = True

    n_samp = env.samp
    batch_size = td['locs'].shape[0]
    encoder_input_size = batch_size // n_samp
    rew_time = 0

    with torch.no_grad():
        with autocast('cuda', dtype=torch.float16):
            node_embeddings, graph_embeddings = encoder(td[:encoder_input_size])
            node_embeddings = node_embeddings.repeat(n_samp, 1, 1)
            graph_embeddings = graph_embeddings.repeat(n_samp, 1)
            if hasattr(decoder, 'build_cache'):
                decoder.build_cache(node_embeddings)
            else:
                decoder.module.build_cache(node_embeddings)

            steps = 0
            while not env.is_done():
                if first and fixed_start is not None:
                    first = False
                    acts = fixed_start
                else:
                    logits = decoder(td, node_embeddings, graph_embeddings)

                    if rollout_type == 'greedy':
                        acts = torch.argmax(logits, dim=1)
                    else:
                        if isinstance(temperature, list):
                            logits = logits / temperature[steps]
                        else:
                            logits = logits / temperature
                        # acts = torch.distributions.Categorical(logits=logits).sample()
                        # probs = torch.softmax(logits, dim=-1)
                        # acts = torch.multinomial(probs, 1).squeeze(-1)

                        gumbel_noise = -torch.empty_like(logits).exponential_().log()  # -log(-log(U))
                        acts = (logits + gumbel_noise).argmax(dim=-1)  # [batch_size]

                td['action'] = acts
                actions.append(acts)
                # print(td['action'].shape)
                td = env.step(td)
                steps += 1
        rew_time_prev = time.time()
        rewards = env.get_reward()
        # print(rewards.mean().item())
        rew_time = time.time() - rew_time_prev
        # print(f'Reward calculation time: {rew_time:.4f} seconds')
        
    # actions = torch.stack(actions).transpose(0, 1).contiguous()
    encoder.train()
    decoder.train()
    # print(rewards.mean().item())

    return rewards, actions, rew_time