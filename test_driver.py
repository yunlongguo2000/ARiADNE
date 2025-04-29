import ray
import numpy as np
import os
import torch

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *

def run_test(run_id, scene_results):
    # make directory for saving gifs, trajectory and length (each run)
    run_gifs_path = os.path.join(test_gifs_path, f'run_{run_id}')
    # print('run_gifs_path:', run_gifs_path)
    run_trajectory_path = os.path.join(test_trajectory_path, f'run_{run_id}')
    run_length_path = os.path.join(test_length_path, f'run_{run_id}')

    if not os.path.exists(run_gifs_path):
        os.makedirs(run_gifs_path)
    if not os.path.exists(run_trajectory_path):
        os.makedirs(run_trajectory_path)
    if not os.path.exists(run_length_path):
        os.makedirs(run_length_path)
        
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    print('Using device:', device)
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    if device.type == 'cuda':
        print('Using GPU')
        checkpoint = torch.load(f'{model_path}/checkpoint.pth')
    else:
        print('Using CPU only')
        checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i, run_gifs_path, run_trajectory_path, run_length_path) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0
    dist_history = []
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
                
                # record the travel distance for each episode
                episode_number = info['episode_number']
                if episode_number not in scene_results:
                    scene_results[episode_number] = []
                scene_results[episode_number].append(metrics['travel_dist'])

            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total test:', NUM_TEST)
        print('|#Average length:', np.array(dist_history).mean())
        print('|#Length std:', np.array(dist_history).std())
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT if NUM_META_AGENT else 0)
class Runner(object):
    def __init__(self, meta_agent_id, run_gifs_path, run_trajectory_path, run_length_path):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)
        self.run_gifs_path = run_gifs_path
        self.run_trajectory_path = run_trajectory_path
        self.run_length_path = run_length_path

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        # make directory for saving gifs, trajectory and length (each episode)
        episode_gifs_path = os.path.join(self.run_gifs_path, f'episode_{episode_number}')
        episode_trajectory_path = os.path.join(self.run_trajectory_path, f'episode_{episode_number}')
        episode_length_path = os.path.join(self.run_length_path, f'episode_{episode_number}')

        if not os.path.exists(episode_gifs_path):
            os.makedirs(episode_gifs_path)
        if not os.path.exists(episode_trajectory_path):
            os.makedirs(episode_trajectory_path)
        if not os.path.exists(episode_length_path):
            os.makedirs(episode_length_path)
        
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number,
                            device=self.device, save_image=SAVE_GIFS, greedy=True,
                            gifs_path=episode_gifs_path, trajectory_path=episode_trajectory_path,
                            length_path=episode_length_path)
        worker.run_episode(episode_number)
        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_weights(weights)
        metrics = self.do_job(episode_number)
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }
        return metrics, info


if __name__ == '__main__':
    ray.init()
    scene_results = {}  # dictionary to store results for each scene
    for i in range(NUM_RUN):
        print(f'Running test round {i + 1}/{NUM_RUN}')
        run_test(i, scene_results)

    # calculate and print the average travel distance for each scene
    print('\n|# Per-scene length statistics:')
    print('|' + '-' * 50)
    for episode_number, lengths in scene_results.items():
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        print(f'| Scene {episode_number}: Mean = {mean_length:.2f}, Std = {std_length:.2f}')