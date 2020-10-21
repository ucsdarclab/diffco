#!/usr/bin/env python
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_commander.conversions import pose_to_list

import threading
import json

import sys
import pickle
sys.path.append('/home/yuheng/FastronPlus-pytorch/')
from Fastronpp import Fastron
from Fastronpp import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
# from Fastronpp.model import RevolutePlanarRobot
# import fcl
# from scipy import ndimage
# from matplotlib import animation
# from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
# import seaborn as sns
# sns.set()
# import matplotlib.patheffects as path_effects
from Fastronpp import utils
# from Fastronpp.Obstacles import FCLObstacle
from Fastronpp.model import BaxterFK


class FastronplusBaxterExperiments(object):
    def __init__(self):
        super(FastronplusBaxterExperiments, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('FastronplusBaxterExperiments', anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "left_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
        robot_state_publisher = rospy.Publisher('/robot/joint_states', JointState, queue_size=1)
        self.pos = [0 for _ in range(7)]
        self.state = JointState()
        self.state.name = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
        self.state.position = self.pos
        self.state_pub_thread = threading.Thread(target=self.pub_state)
        
        
        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.robot_state_publisher = robot_state_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.state_pub_thread.start()
    
    def pub_state(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.state.position = self.pos
            self.state.header.stamp = rospy.Time.now()
            self.robot_state_publisher.publish(self.state)
            rate.sleep()
        print('***************************************************State Publisher Exited!!!')

    def go_to_joint_state(self, joint_pos):
        self.pos = joint_pos


    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

        ## END_SUB_TUTORIAL


    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Received
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL


    def add_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL add_box
        ##
        ## Adding Objects to the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## First, we will create a box in the planning scene at the location of the left finger:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "baxter_leftfinger"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.07 # slightly above the end effector
        box_name = "box"
        scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))

        ## END_SUB_TUTORIAL
        # Copy local variables back to class variables. In practice, you should use the class
        # variables directly unless you have a good reason not to.
        self.box_name=box_name
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)


    def remove_box(self, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL remove_object
        ##
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(box_name)

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

def wait_for_state_update(scene, box_name, box_is_known=False, box_is_attached=False, timeout=4):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
        # Test if the box is in attached objects
        attached_objects = scene.get_attached_objects([box_name])
        is_attached = len(attached_objects.keys()) > 0

        # Test if the box is in the scene.
        # Note that attaching the box will remove it from known_objects
        is_known = box_name in scene.get_known_object_names()

        # Test if we are in the expected state
        if (box_is_attached == is_attached) and (box_is_known == is_known):
            return True

        # Sleep so that we give other threads time on the processor
        rospy.sleep(0.1)
        seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False

def traj_optimize(robot, start_cfg, target_cfg, dist_est, initial_guess=None, history=False):
    N_WAYPOINTS = 50
    NUM_RE_TRIALS = 1 #10
    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 40
    collision_weight = 20
    joint_limit_weight = 20
    safety_margin = -1.0
    lr = 1e-2
    seed = 19961221
    torch.manual_seed(seed)

    lowest_cost_solution = None
    lowest_cost = np.inf
    lowest_cost_trial = None
    lowest_cost_step = None
    best_valid_solution = None
    best_valid_cost = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []

    found = False
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            if initial_guess is None:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
            else:
                init_path = initial_guess
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)

        for step in range(UPDATE_STEPS):
            opt.zero_grad()
            assert p.dtype == torch.float
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            # max_move_cost = torch.clamp(
            #     (p[1:]-p[:-1]).pow(2).sum(dim=1)-0.3**2, min=0).sum() # for rbf kernel
            control_points = robot.fkine(p, reuse=True)
            max_move_cost = torch.clamp(
                (control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-0.03**2, min=0).sum()
                # (control_points[1:, -2:-1]-control_points[:-1, -2:-1]).pow(2).sum(dim=2)-0.03**2, min=0).sum()
            joint_limit_cost = (
                torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum() + (utils.wrap2pi(p[1:]-p[:-1])).pow(2).sum()

            constraint_loss = collision_weight * collision_score \
                + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            opt_loss = dif_weight * diff
            loss = constraint_loss + opt_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data = utils.wrap2pi(p.data)
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_cost:
                lowest_cost = loss.data.numpy()
                lowest_cost_solution = p.data.clone()
                lowest_cost_step = step
                lowest_cost_trial = trial_time
            if constraint_loss <= 1e-2:
                if opt_loss.data.numpy() < best_valid_cost:
                    best_valid_cost = opt_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
                print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, joint_limit={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    joint_limit_cost.item(), joint_limit_weight,
                    diff.item(), dif_weight,
                    loss.item()))
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    
    if not found:
        print('Did not find a valid solution after {} trials!\
            Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_cost_solution
        solution_step = lowest_cost_step
        solution_trial = lowest_cost_trial
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    return solution, path_history, solution_trial, solution_step

def animate_path(p, obstacles):
    exp = FastronplusBaxterExperiments()
    print('Adding box')
    rospy.sleep(2)

    box_names = []
    for i, obs in enumerate(obstacles):
        box_name = 'box_{}'.format(i)
        box_names.append(box_name)
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = exp.planning_frame
        # box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = obs[1][0]
        box_pose.pose.position.y = obs[1][1]
        box_pose.pose.position.z = obs[1][2]
        exp.scene.add_box(box_name, box_pose, size=obs[2])
        wait_for_state_update(exp.scene, box_name, box_is_known=True)

    waypoint_idx = 0
    rate = rospy.Rate(3)
    while not rospy.is_shutdown():
        waypoint_idx = waypoint_idx % len(p)
        exp.go_to_joint_state(p[waypoint_idx])
        waypoint_idx += 1
        rate.sleep()
    
    return box_names, exp

def single_shot(path, obstacles):
    exp = FastronplusBaxterExperiments()
    print('Adding box')
    rospy.sleep(2)

    box_names = []
    # for i, obs in enumerate(obstacles):
    #     box_name = 'box_{}'.format(i)
    #     box_names.append(box_name)
    #     box_pose = geometry_msgs.msg.PoseStamped()
    #     box_pose.header.frame_id = exp.planning_frame
    #     # box_pose.pose.orientation.w = 1.0
    #     box_pose.pose.position.x = obs[1][0]
    #     box_pose.pose.position.y = obs[1][1]
    #     box_pose.pose.position.z = obs[1][2]
    #     exp.scene.add_box(box_name, box_pose, size=obs[2])
    #     wait_for_state_update(exp.scene, box_name, box_is_known=True)
    
    pub = exp.display_trajectory_publisher
    
    joint_traj = JointTrajectory()
    for q in path:
        traj_point = JointTrajectoryPoint()
        traj_point.positions = q.numpy().tolist()
        joint_traj.points.append(traj_point)
    joint_traj.joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory = joint_traj
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_traj)
    disp_traj.trajectory_start.joint_state.position = path[0].numpy()
    disp_traj.trajectory_start.joint_state.name = joint_traj.joint_names
    pub.publish(disp_traj)
    
    return box_names, exp

def escape(robot, dist_est, start_cfg):
    N_WAYPOINTS = 20
    # NUM_RE_TRIALS = 10
    UPDATE_STEPS = 200
    # dif_weight = 1
    # max_move_weight = 10
    # collision_weight = 10
    safety_margin = -5 #torch.FloatTensor([-2, -0.2])
    lr = 5e-2
    # seed = 19961221
    # torch.manual_seed(seed)

    # lowest_cost_solution = None
    # lowest_cost = np.inf
    # lowest_cost_trial = None
    # lowest_cost_step = None
    # best_valid_solution = None
    # best_valid_cost = np.inf
    # best_valid_step = None
    # best_valid_trial = None
    
    # trial_histories = []

    # found = False
    # p = torch.FloatTensor(np.concatenate([np.linspace(start_cfg, (-np.pi, 0), N_STEPS/2), np.linspace((np.pi, 0), target_cfg, N_STEPS/2)], axis=0)).requires_grad_(True)
    # for trial_time in range(NUM_RE_TRIALS):
    path_history = []
    # if trial_time == 0:
    #     init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=UPDATE_STEPS))
    # else:
    #     init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
    init_path = start_cfg
    # init_path[-1] = target_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    # opt = torch.optim.SGD([p], lr=lr, momentum=0.0)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())

        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin #, min=0).sum()
        # print(torch.clamp(dist_est(p)-safety_margin, min=0).max(dim=0).values.data)
        # control_points = robot.fkine(p)
        # max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-1.0**2, min=0).sum()
        # diff = dif_weight * (control_points[1:]-control_points[:-1]).pow(2).sum()
        # np.clip(1.5*float(i)/UPDATE_STEPS, 0, 1)**2 (float(i)/UPDATE_STEPS) * 
        # torch.clamp(utils.wrap2pi(p[1:]-p[:-1]).abs(), min=0.3).pow(2).sum()
        # constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
        # objective_loss = dif_weight * diff
        loss = collision_score #objective_loss + constraint_loss
        loss.backward()
        # p.grad[[0, -1]] = 0.0
        opt.step()
        p.data = utils.wrap2pi(p.data)
        # if history:
        
        # if loss.data.numpy() < lowest_cost:
        #     lowest_cost = loss.data.numpy()
        #     lowest_cost_solution = p.data.clone()
        #     lowest_cost_step = step
        #     lowest_cost_trial = trial_time
        if collision_score <= 1e-4:
            # if objective_loss.data.numpy() < best_valid_cost:
            #     best_valid_cost = objective_loss.data.numpy()
            #     best_valid_solution = p.data.clone()
            #     best_valid_step = step
            #     best_valid_trial = trial_time
            break
        # if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
        #     print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
        #         trial_time, step, 
        #         collision_score.item(), collision_weight,
        #         max_move_cost.item(), max_move_weight,
        #         diff.item(), dif_weight,
        #         loss.item()))
        # trial_histories.append(path_history)
        
        # if best_valid_solution is not None:
        #     found = True
        #     break
    # if not found:
    #     print('Did not find a valid solution after {} trials!\
    #         Giving the lowest cost solution'.format(NUM_RE_TRIALS))
    #     solution = lowest_cost_solution
    #     solution_step = lowest_cost_step
    #     solution_trial = lowest_cost_trial
    # else:
    #     solution = best_valid_solution
    #     solution_step = best_valid_step
    #     solution_trial = best_valid_trial
    # path_history = trial_histories[solution_trial] # Could be empty when history = false
    # if not path_history:
    #     path_history.append(solution)
    # else:
    #     path_history = path_history[:(solution_step+1)]
    return torch.stack(path_history, dim=0)# sum(trial_histories, []),


def main():
    robot_name = 'baxter'
    env_name = 'medium' #'complex' # 2objontable' # 'table'
    DOF = 7

    dataset = torch.load('data/3d_{}_{}.pt'.format(robot_name, env_name))
    cfgs = dataset['data']
    cfgs = cfgs.type(torch.float)
    labels = dataset['label']
    # dists = dataset['dist']
    obstacles = dataset['obs']
    robot = dataset['robot']()
    train_num = int(len(cfgs) * 0.9)
    fkine = robot.fkine
    '''
    #====
    checker = Fastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(500)), beta=1.0)
    # checker = Fastron(obstacles, beta=1.0)
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]))
    with open('results/checker.p', 'wb') as f:
        pickle.dump(checker, f)
        print('checker saved')
    #====
    with open('results/checker.p', 'rb') as f:
        checker = pickle.load(f)

    # Check Fastron test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds)
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    assert(test_acc > 0.8)

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_rbf(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine) # epsilon=Epsilon,
    # checker.fit_poly(epsilon=Epsilon, target=fitting_target, fkine=fkine)
    dist_est = checker.rbf_score
    # spline_func = checker.poly_score
    MIN_SCORE = dist_est(cfgs[train_num:]).min().item()
    print('MIN_SCORE = {}'.format(MIN_SCORE))

    free_cfgs = cfgs[labels == -1]
    indices = torch.randint(0, len(free_cfgs), (2, ))
    while indices[0] == indices[1]:
        indices = torch.randint(0, len(free_cfgs), (2, ))
    # start_cfg = torch.FloatTensor([-39, 40, -111, 81, 4, 29, -136])/180*pi # free_cfgs[indices[0]]
    start_cfg = torch.FloatTensor([25, 31, -120, 58, -66, -8, 116])/180*pi # medium scene
    # torch.FloatTensor([5, 51, -126, 58, -66, -8, 116])/180*pi # complex scene, start below objects
    # torch.FloatTensor([-5, 49, -146, 41, -107, -20, 168])/180*pi #2obj scene, start from left side of the objects
    # torch.FloatTensor([-41, 27, -88, 23, -166, 67, 168])/180*pi # 2obj scene, start from between the objs
    # torch.FloatTensor([-27, 34, -92, 34, -174, -50, -19]) /180*pi #start from high-risk (basic scene)
    target_cfg = torch.FloatTensor([-48, 59, -147, 50, -170, -30, 169])/180*pi #complex scene, end below objects; medium scene
    # target_cfg = torch.FloatTensor([-22, 28, -87, 45, -170, -30, 169])/180*pi #2obj scene, stop on the right of the objects
    # torch.FloatTensor([13, 31, -88, 16, -160, -27, 169])/180*pi # 2obj scene, stop beside table
    # torch.FloatTensor([4, 29, -86, 44, 3, 16, -146])/180*pi
    
    with open('data/medium_success_2.json', 'r') as f:
        init_guess = torch.FloatTensor(json.load(f)['path'])
    p, path_history, num_trial, num_step = traj_optimize(robot, start_cfg, target_cfg, dist_est, init_guess, history=True)

    # p, path_history, num_trial, num_step = traj_optimize(robot, start_cfg, target_cfg, dist_est, history=True)
    with open('results/path_3d_{}_{}.json'.format(robot_name, env_name), 'w') as f:
        json.dump(
            {
                'path': p.data.numpy().tolist(), 
                'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
                'trial': num_trial,
                'step': num_step
            },
            f, indent=1)
        print('Plan recorded in {}'.format(f.name))
    # p = escape(robot, dist_est, start_cfg)
    '''

    # with open('results/path_3d_{}_{}.json'.format(robot_name, env_name), 'r') as f:
    #     p = torch.FloatTensor(json.load(f)['path'])
    with open('data/medium_success_2.json', 'r') as f:
        p = torch.FloatTensor(json.load(f)['path'])

    try:
        raw_input = input
    except NameError:
        pass
    try:
        print("Press Ctrl-D to exit at any time")
        print("")
        print("============ Press `Enter` to visualize the path generated")
        raw_input()

        box_names = []
        # box_names, exp = animate_path(p, obstacles)
        box_names, exp = single_shot(p, obstacles)

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    
    for box_name in box_names:
        exp.scene.remove_world_object(box_name)
        print('Tried removing {}!'.format(box_name))
        wait_for_state_update(exp.scene, box_name, box_is_attached=False, box_is_known=False)

if __name__ == "__main__":
    main()


