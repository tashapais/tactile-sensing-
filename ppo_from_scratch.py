import torch
import argparse
import random
import numpy as np
import gym
from grid_world_env import GridWorldEnv
from ppo_discrete import VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import tqdm
import misc_utils as mu
from distutils.util import strtobool
import os, wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument('--multiprocess', type=lambda x: bool(strtobool(x)), 
                        default=True, nargs='?', const=True)
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument('--all_in_one', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='all in one policy: the agent output everything and discriminator is not needed')
    parser.add_argument('--collect_initial_batch', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='whether we collect an initial batch of data to train the discriminator before updating the explorer')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args

def make_env(seed, idx):
    def thunk():
        env = GridWorldEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk




def train_discriminator(discriminator,
                        dataset, 
                        agent, 
                        discriminator_data_batch, 
                        update,
                        explore_updates,
                        next_obs,
                        logger,
                        writer):
    if args.train_discriminator:
        if args.collect_initial_batch and discriminator_data_batch == 0:
            # collect and train on first batch of data
            pbar = tqdm.tqdm(total=dataset.buffer_size)
            while len(dataset) < dataset.buffer_size:
                # TODO further verify list of dict or dict of list: look at source code, infos is a tuple of dicts
                # using the initialized policy instead of the random policy to collect data
                if args.initial_batch_policy == 'random':
                    move = [random.choice([0, 1, 2, 3]) for i in range(args.num_envs)]
                    current_steps = envs.get_attr('current_step')
                    done = [1 if current_steps[i] == args.initial_batch_ep_len else 0 for i in range(args.num_envs)]
                elif args.initial_batch_policy == 'agent':
                    move = agent.get_move(next_obs)[0]
                    move = [i.item() for i in move]
                    current_steps = envs.get_attr('current_step')
                    done = [1 if current_steps[i] == args.initial_batch_ep_len else 0 for i in range(args.num_envs)]
                else:
                    raise TypeError('unrecognized initial batch policy')
                action = [{'move': move[i],
                           'prediction': 0,
                           'done': done[i],
                           'max_prob': 0.1,
                           'probs': [0.1] * 10}
                          for i in range(args.num_envs)]
                next_obs, rs, ds, infos = envs.step(action)
                for (i, info, done) in zip(range(len(infos)), infos, ds):
                    if info['discover']:
                        # only next_obs is from the reset of the next episode and reset only returns obs
                        # without info
                        imgs = mu.generate_rotated_imgs(mu.get_discriminator_input(info['ob']),
                                                        num_rotations=args.num_rotations)
                        # imgs = mu.rotate_imgs(imgs, [-info['angle']])
                        dataset.add_data(imgs,
                                                [info['num_gt']] * args.num_rotations)
                        pbar.update(args.num_rotations)
                    if len(dataset) == dataset.buffer_size:
                        break
            pbar.close()
            # reset the next_obs so that the RL training does not start with highly revealed observations from the random policy
            next_obs = envs.reset()
            return next_obs
        if len(dataset) >= dataset.buffer_size and (update - 1) % explore_updates == 0:
            # train discriminator
            logger.log(str(len(dataset)))
            logger.log(f'discriminator data batch: {discriminator_data_batch}')
            folder_name = f'discriminator_batch_{discriminator_data_batch:04d}'
            pixel_freq = mu.compute_pixel_freq(dataset.imgs, visualize=False, save=True,
                                               save_path=os.path.join(args.save_dir, folder_name, 'data',
                                                                      'pixel_freq.png'))
            # set path for learning to save checkpoint
            discriminator.save_dir = os.path.join(args.save_dir, folder_name)
            if args.save_discriminator_data:
                dataset.export_data(os.path.join(args.save_dir, folder_name, 'data'))
            train_loader, test_loader = mu.construct_loaders(dataset=dataset, split=0.2)
            # always train 15 epochs for the first discriminator
            discriminator_path, discriminator_train_loss, discriminator_train_acc, discriminator_test_loss, discriminator_test_acc, stats = discriminator.learn(
                epochs=15 if discriminator_data_batch == 0 else args.discriminator_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                logger=logger)

            # write discriminator stats
            for i, stat in enumerate(stats):
                writer.add_scalar('discriminator/train_loss', stat['train_loss'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/train_acc', stat['train_acc'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/test_loss', stat['test_loss'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/test_acc', stat['test_acc'],
                                  discriminator_data_batch * args.discriminator_epochs + i)

                if args.prod_mode:
                    data_to_log = {
                        'discriminator/train_loss': stat['train_loss'],
                        'discriminator/train_acc': stat['train_acc'],
                        'discriminator/test_loss': stat['test_loss'],
                        'discriminator/test_acc': stat['test_acc'],
                        'discriminator_batch': discriminator_data_batch * args.discriminator_epochs + i,
                    }
                    wandb.log(data_to_log)
            discriminator_data_batch += 1
            
            return (discriminator_path, discriminator_train_loss, discriminator_train_acc, discriminator_test_loss, discriminator_test_acc)


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.all_in_one:
        pass

    if args.multiprocess:
        envs = VecPyTorch(
            SubprocVecEnv([make_env(args.seed + i, i)
                           for i in range(args.num_envs)], "fork"), device)
    else:
        envs = VecPyTorch(
            DummyVecEnv([make_env(args.seed + i, i)
                         for i in range(args.num_envs)]), device)
        


