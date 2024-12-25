import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import copy

class Environment:
    def __init__(self):
        self.node_coords = np.array([[100, 200], [300, 400], [500, 600]])
        self.graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        self.node_utility = np.array([10, 20, 30])
        self.guidepost = np.array([[1, 0], [0, 1], [1, 1]])

    def find_index_from_coords(self, coords):
        for i, node in enumerate(self.node_coords):
            if np.array_equal(node, coords):
                return i
        return -1

class Robot:
    def __init__(self, env):
        self.env = env
        self.robot_position = np.array([500, 600])
        self.node_padding_size = 5
        self.k_size = 3
        self.device = torch.device('cpu')

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # padding the number of node to a given node padding size
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(0)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        # calculate a mask for the padded edges (denoted by 0)
        edge_padding_mask = torch.zeros((1, 1, self.k_size), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        mask = np.zeros((size, size))
        for i, edges in enumerate(edge_inputs):
            for edge in edges:
                mask[i][edge] = 1
        return mask

# 创建环境和机器人实例
env = Environment()
robot = Robot(env)

# 获取观测数据
observations = robot.get_observations()

# 可视化节点和边
G = nx.Graph()
for i, coord in enumerate(env.node_coords):
    G.add_node(i, pos=(coord[0], coord[1]))

for node, edges in env.graph.items():
    for edge in edges:
        G.add_edge(node, edge)

pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, font_color='black')
plt.title("Node and Edge Visualization")
plt.show()