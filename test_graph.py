import matplotlib.pyplot as plt
import numpy as np
from graph_generator import Graph_generator

def visualize_graph(graph_generator, robot_location, robot_belief, frontiers):
    # 生成图结构
    node_coords, edges, node_utility, guidepost = graph_generator.generate_graph(robot_location, robot_belief, frontiers)

    # 调试输出
    print("Node Coordinates:", node_coords)
    print("Edges:", edges)

    # 绘制节点
    plt.scatter(node_coords[:, 0], node_coords[:, 1], c='blue', label='Nodes')

    # 绘制边
    for start_node, connections in edges.items():
        for end_node in connections:
            start_node_coords = node_coords[int(start_node)]
            end_node_coords = node_coords[int(end_node)]
            plt.plot([start_node_coords[0], end_node_coords[0]], [start_node_coords[1], end_node_coords[1]], 'k-', lw=0.5)

    # 绘制机器人位置
    plt.scatter(robot_location[0], robot_location[1], c='red', label='Robot Location')

    # 绘制前沿点
    plt.scatter(frontiers[:, 0], frontiers[:, 1], c='green', label='Frontiers')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph Structure Visualization')
    plt.legend()
    plt.show()

# 示例使用
map_size = (100, 100)
k_size = 5
sensor_range = 10
robot_location = np.array([50, 50])
robot_belief = np.zeros(map_size)
robot_belief[30:70, 30:70] = 255  # 模拟自由区域
frontiers = np.array([[40, 40], [60, 60], [50, 70]])

graph_generator = Graph_generator(map_size, k_size, sensor_range)
visualize_graph(graph_generator, robot_location, robot_belief, frontiers)