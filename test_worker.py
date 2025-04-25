import os
import csv
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from env import Env
from utils import *
from model import PolicyNet
from agent import Agent
from test_parameter import *  # 包含：MAX_EPISODE_STEP, SAVE_TRAJECTORY, trajectory_path, SAVE_LENGTH, length_path, gifs_path

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image

        # 使用新版本的环境和智能体
        print("Gifs path:", gifs_path)
        self.env = Env(global_step, plot=save_image, gifs_path=gifs_path)
        self.robot = Agent(policy_net, self.device, save_image)

        self.travel_dist = 0
        self.perf_metrics = dict()
        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self, curr_episode):
        done = False
        # 更新规划状态：更新地图、当前位置、frontier等
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            # 保存轨迹（若配置要求）
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = os.path.join('results', 'trajectory', 'ours_trajectory_result.csv')
                new_file = not os.path.exists(csv_filename)
                field_names = ['dist', 'area']
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.env.travel_dist, np.sum(self.env.robot_belief == 255)]).reshape(1, -1)
                    writer.writerows(csv_data)

            self.save_observation(observation)
            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            # 检查下一位置是否在候选邻居中
            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1]*1j in (check[:,0] + check[:,1]*1j), \
                print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            self.save_next_observations(observation)

            if self.save_image:
                self.robot.plot_env()
                self.env.plot_env(i+1)

            if done:
                break

        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = os.path.join('results', 'length', 'ours_length_result.csv')
            new_file = not os.path.exists(csv_filename)
            field_names = ['dist']
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.env.travel_dist]).reshape(-1, 1)
                writer.writerows(csv_data)

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()