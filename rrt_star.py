import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mlxtend.plotting import plot_decision_regions
from Fastron import *
from MultiFastron import *

def euclidean_dist(a, b):
    return np.linalg.norm(a-b)

def radius_local_planner(start, target, dist_func, radius=0.5, n_local_plan=10):
    if dist_func(start, target) < 0.5:
        return target
    thetas = np.arange(0, 2*np.pi, 2*np.pi / n_local_plan)
    # print('Thetas: ', thetas)
    local_targets = start + np.stack([np.cos(thetas)*radius, np.sin(thetas)*radius], axis=1)
    distances = np.fromiter(map(lambda config: dist_func(config, target), local_targets), float)
    # print('Distances: ', distances)
    # print('min move: ', local_targets[np.argmin(distances)])
    return local_targets[np.argmin(distances)]


class RRT_STAR:
    class Node:
        def __init__(self, config, dist=np.inf, parent=None):
            self.config = config
            self.parent = parent if parent is not None else self
            self.dist = dist

    def __init__(self, start, goal, space_range, obstacles, local_planner, collision_checker=None, dist_func=euclidean_dist, rewire=True):
        self.start = start
        self.goal = goal
        self.space_range = space_range
        self.obstacles = obstacles
        self.local_planner = local_planner
        self.dist_func = dist_func
        self.rewire = rewire
        self.node_list = []
        self.figure, self.ax = None, None
        self.route = None
        self.end_node = None
        self.shortest_dist = np.inf
        self.collision_checker = CollisionChecker(self.obstacles) if collision_checker is None else collision_checker
        self.animate_cnt = 0

    def weighted_dist(self, a, b):
        pred_a, pred_b = self.collision_checker.predict(a), self.collision_checker.predict(b)
        cost_a, cost_b = self.obstacles[pred_a-1].get_cost() if pred_a > 0 else 0, self.obstacles[pred_b-1].get_cost() if pred_b > 0 else 0
        return np.linalg.norm(a-b)*(1+np.maximum(cost_a, cost_b)) if np.maximum(cost_a, cost_b) != np.inf else np.inf
        
    
    def plan(self, max_tree_size=10000, animate_interval=50):
        start_node = self.Node(self.start, dist=0)
        self.node_list.append(start_node)
        while len(self.node_list) < max_tree_size:
            if len(self.node_list) % animate_interval == 0:
                print('Animate')
                self.route, self.shortest_dist = self.get_route()
                self.animate(self.route)
            if np.random.rand() < 0.1: 
                sample = np.array(self.goal)
            else:
                sample = np.random.rand(2) * self.space_range
            
            distances = np.array(list(map(lambda node: self.dist_func(node.config, sample), self.node_list)))
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]
            nearest_node = self.node_list[nearest_idx]
            new_state = self.local_planner(nearest_node.config, sample, self.dist_func)
            # print('nearest distance, ', nearest_dist, sample, nearest_node.config, new_state)
            # print(new_state, len(self.node_list))
            
            # collision_checks = map(lambda obs: obs.is_collision(new_state), self.obstacles)
            # print(new_state, self.collision_checker.line_collision(nearest_node.config, new_state))
            if not self.collision_checker.line_collision(nearest_node.config, new_state):
                new_node = self.Node(new_state, dist=nearest_node.dist+self.dist_func(nearest_node.config, new_state), parent=nearest_node)
                if self.dist_func(new_state, self.goal) < 1e-4 and new_node.dist < self.shortest_dist:
                    self.route, self.shortest_dist = self.get_route()
                    if new_node.dist < self.shortest_dist:
                        self.end_node = new_node
                        self.route, self.shortest_dist = self.get_route()
                        print('Dist: ', self.shortest_dist)
                        # traj = []
                        # t = new_node
                        # check_dist = 0
                        # while t.parent is not t:
                        #     traj.append(t.config)
                        #     check_dist += self.dist_func(t.config, t.parent.config)
                        #     t = t.parent
                        # traj.append(t.config)
                        # print('Check dist:', check_dist)
                        # # traj = traj[::-1]
                        # self.route = traj
                        
                if self.rewire and len(self.node_list) > 20:
                    new_distances = np.array(list(map(lambda node: self.dist_func(node.config, new_state), self.node_list)))
                    near_idx = filter(lambda i: new_distances[i] < 3, range(len(new_distances)))
                    # print('near_idx:', list(near_idx))
                    # nearest_k_idx = np.argpartition(new_distances, 20)
                    for idx in near_idx:
                        to_new_distance = new_distances[idx]

                        if not self.collision_checker.line_collision(new_state, self.node_list[idx].config):
                            tmp_dist = new_node.dist + to_new_distance
                            if tmp_dist < self.node_list[idx].dist:
                                # print(tmp_dist, self.node_list[idx].dist)
                                self.node_list[idx].dist = tmp_dist
                                self.node_list[idx].parent = new_node
                                # self.node_list[idx].parent = start_node
                self.node_list.append(new_node)
        print('Finalized')
        self.route, self.shortest_dist = self.get_route()
        self.animate(self.route)
        return self.route
    
    def get_route(self):
        if self.end_node is None:
            return [], np.inf
        else:
            traj = []
            t = self.end_node
            check_dist = 0
            while t.parent is not t:
                traj.append(t.config)
                check_dist += self.dist_func(t.parent.config, t.config)
                t = t.parent
            traj.append(t.config)
            print('Min Dist = ', self.shortest_dist, 'Check dist = ', check_dist)
            traj = traj[::-1]
            return traj, check_dist
    
    def animate(self, traj=None):
        self.animate_cnt += 1
        segments = map(lambda node: (node.config, node.parent.config), self.node_list)
        line_segments = LineCollection(segments)
        _, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 15)
        self.ax.set_ylim(-5, 15)
        self.ax.set_aspect('equal', 'datalim')
        self.ax.add_collection(line_segments)
        plt.scatter(self.start[0], self.start[1], linewidths=0.5, marker='o', c='r')
        plt.scatter(self.goal[0], self.goal[1], linewidths=0.5, marker='o', c='g')
        if traj is not None:
            xs = list(map(lambda v: v[0], traj))
            ys = list(map(lambda v: v[1], traj))
            plt.plot(xs, ys, color='y', linestyle='--')
        for obs in self.obstacles:
            if obs.kind == 'circle':
                circle_artist = plt.Circle(obs.position, radius=obs.size/2, color='black')
                self.ax.add_artist(circle_artist)
            elif obs.kind == 'rect':
                rect_artist = plt.Rectangle(obs.position-obs.size/2, obs.size[0], obs.size[1], color='black')
                self.ax.add_artist(rect_artist)
            else:
                raise NotImplementedError('Unknown obstacle type')
        plt.show()
        


if __name__ == '__main__':
    obstacles = [
        ('circle', (2, 1), 3, 0.2),
        ('rect', (5, 7), (5, 3), np.inf)]
    obstacles = [Obstacle(*param) for param in obstacles]
    print(obstacles)
    # obstacles = []
    checker = MultiFastron(obstacles, len(obstacles))
    checker.train(1000)
    checker.vis()
    planner = RRT_STAR((0, 0), (10, 10), (10, 10), obstacles, radius_local_planner)#, euclidean_dist, rewire=True)
    planner.collision_checker = checker
    planner.dist_func = planner.weighted_dist
    print(planner.plan(200, animate_interval=50))
    # checker = Fastron(obstacles)
    
    # plt.axis('equal')
    # plt.xlim((0, 10))
    # plt.ylim((0, 10))
    # plot_decision_regions(X=checker.support_points, y=checker.y.astype(np.integer), clf=checker, markers=[None], legend=None, filler_feature_ranges=[(0, 10), (0, 10)])
    # plt.show()
    # print(checker.support_points, checker.gains)
    # checker.vis()
            

