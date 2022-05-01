import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac, a3c
from ray.rllib.agents.marwil.bc import BCTrainer, BC_DEFAULT_CONFIG
from ray.rllib.agents.marwil.marwil import MARWILTrainer
from ray.rllib.agents.marwil.marwil import DEFAULT_CONFIG as MARWIL_DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from numpngw import write_apng
from ray import tune

def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.99
        config['model']['fcnet_hiddens'] = [100, 100]
        config['lr'] = 5e-5 # default = 5e-5
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False
    elif algo == 'a3c':
        config = a3c.DEFAULT_CONFIG.copy()
        config["use_critic"] = True
        config["use_gae"] = True
        # Size of rollout batch
        config["rollout_fragment_length"] = 20
        config["grad_clip"] = 40.0
        config["lr"] = 1e-7
        config["lr_schedule"] = None
        config["vf_loss_coeff"] = 0.75
        config["entropy_coeff"] = 0.0
        config["entropy_coeff_schedule"] = None
        # config["min_time_s_per_reporting"] = 5
        # Workers sample async. Note that this increases the effective
        # rollout_fragment_length by up to 5x due to async buffering of batches.
        config["sample_async"] = False
        # config["_disable_execution_plan_api"] = True

    elif algo == 'bc':
        config = BC_DEFAULT_CONFIG.copy()
        config['model']['vf_share_layers'] = False
        config['model']['fcnet_hiddens'] = [32,16]

        config["lr"]: 1e-4

        # config["lr"]: tune.grid_search([0.01, 0.001, 0.0001])

        # expert demonstration
        # config["input"] = "/home/hrl_gpu_1/hrl_git/assistive_gym_fem_tri/examples/demo-out/"
        config['input'] = "sampler"
        config["input_config"]={
            "format": "json",  # json or parquet
            # Path to data file or directory.
            "path": "/home/hrl_gpu_1/hrl_git/assistive_gym_fem_tri/examples/",
            # Num of tasks reading dataset in parallel, default is num_workers.
            # "parallelism": 3,
            # Dataset allocates 0.5 CPU for each reader by default.
            # Adjust this value based on the size of your offline dataset.
            # "num_cpus_per_read_task": 0.5,
        }
        config["use_gae"] = True
        # When beta is 0.0, MARWIL is reduced to behavior cloning
        config["beta"] = 0.0
        # Balancing value estimation loss and policy optimization loss.
        config["vf_coeff"] = 1.0
        # If specified, clip the global norm of gradients by this amount.
        config["grad_clip"] = None
        # config["lr"] = (tune.loguniform(1e-4, 1e-2))
        # The squared moving avg. advantage norm (c^2) update rate
        # (1e-8 in the paper).
        config["moving_average_sqd_adv_norm_update_rate"] = 1e-8
        # Starting value for the squared moving avg. advantage norm (c^2).
        config["moving_average_sqd_adv_norm_start"] = 100.0
        # Number of (independent) timesteps pushed through the loss
        # each SGD round.
        config["train_batch_size"] = 2000
        # config["train_batch_size"] = 1000
        # Size of the replay buffer in (single and independent) timesteps.
        # The buffer gets filled by reading from the input files line-by-line
        # and adding all timesteps on one line at once. We then sample
        # uniformly from the buffer (`train_batch_size` samples) for
        # each training step.
        # config["replay_buffer_size"] = 10000
        # config["replay_buffer_size"] = 150
        # Number of steps to read before learning starts.
        config["learning_starts"] = 0
    
    elif algo == 'marwil':
        config = MARWIL_DEFAULT_CONFIG.copy()
        # expert demonstration
        # config["input"] = "/home/hrl_gpu_1/hrl_git/assistive_gym_fem_tri/examples/demo-out/"
        config['input'] = "sampler"
        config["input_config"]={
            "format": "json",  # json or parquet
            # Path to data file or directory.
            "path": "/home/hrl_gpu_1/hrl_git/assistive_gym_fem_tri/examples/",
            # "parallelism": 3,
            # "num_cpus_per_read_task": 0.5,
        }
        config["use_gae"] = True
        # When beta is 0.0, MARWIL is reduced to behavior cloning
        config["beta"] = 1.0
        # Balancing value estimation loss and policy optimization loss.
        config["vf_coeff"] = 1.0
        # If specified, clip the global norm of gradients by this amount.
        config["grad_clip"] = None
        # Learning rate for Adam optimizer.
        config["lr"] = 1e-4
        # The squared moving avg. advantage norm (c^2) update rate
        # (1e-8 in the paper).
        config["moving_average_sqd_adv_norm_update_rate"] = 1e-8
        # Starting value for the squared moving avg. advantage norm (c^2).
        config["moving_average_sqd_adv_norm_start"] = 100.0
        # Number of (independent) timesteps pushed through the loss
        # each SGD round.
        config["train_batch_size"] = 2000
        # config["train_batch_size"] = 150
        # Size of the replay buffer in (single and independent) timesteps.
        # The buffer gets filled by reading from the input files line-by-line
        # and adding all timesteps on one line at once. We then sample
        # uniformly from the buffer (`train_batch_size` samples) for
        # each training step.
        config["replay_buffer_size"] = 10000
        # config["replay_buffer_size"] = 450
        # Number of steps to read before learning starts.
        config["learning_starts"] = 0
        
    config['num_workers'] = num_processes // 2
    # config['num_workers'] = 1
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    # if algo == 'sac':
    #     config['num_workers'] = 1
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'a3c':
        agent = a3c.A2CTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'bc':
        agent = BCTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'marwil':
        agent = MARWILTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)

    print(policy_path)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            print(files)
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                print(checkpoint_path)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs)
    env.disconnect()

    timesteps = 0
    while timesteps < timesteps_total:
        # result = dict()
        result = agent.train()
        # analysis = tune.run("BC", config=setup_config(env_name, algo, coop, seed, extra_configs))

        timesteps = 0
        timesteps = result['timesteps_total']

        # result['episode_reward_mean'] = analysis.get_best_trial("episode_reward_mean")
        # result['episode_reward_min'] = analysis.best_result
        # result['episode_reward_max'] = analysis.best_result

        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2

        # data to graph
        reward_path = './rewards/'
        if not os.path.exists(reward_path):
            os.makedirs(reward_path)
        f = open(reward_path + algo + '_rewards' + '.txt', 'a')
        f.write(str([timesteps, result['episode_reward_mean']]) + ', ')
        f.close()
        
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))
    return checkpoint_path

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    if not colab:
        env.render()
    # robot_action_array = []
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # human_action_array.append(action_human)
                # robot_action_array.append(action_robot)
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
    env.disconnect()
    
    # np_human_actions = np.asarray(human_action_array, dtype=np.float32)
    # #np.save('./trained_models/human_actions_sample.npy', np_human_actions)

    # np_robot_actions = np.asarray(robot_action_array, dtype=np.float32)
    # np.save('./trained_models/robot_actions_sample.npy', np_robot_actions) 
    
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename

def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    rewards = []
    forces = []
    task_successes = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
            reward_total += reward
            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    env.disconnect()

    print('\n', '-'*50, '\n')
    print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    print('Task_successes total', np.sum(task_successes))
    print('Task_successes total', task_successes)
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)

