#!/usr/bin/env python
import copy

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import dist, pi
from std_msgs.msg import String, Empty
from sensor_msgs.msg import JointState
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_commander.conversions import pose_to_list

import threading
import json
from time import time

import sys
import pickle
from diffco import DiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco import utils
from diffco.model import BaxterLeftArmFK


class DiffCoplusBaxterExperiments(object):
    def __init__(self):
        super(DiffCoplusBaxterExperiments, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('DiffCoplusBaxterExperiments', anonymous=True)

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
        group_name =  'both_arms' # 'panda_arm' # baxter: "left_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory)#,
                                                    # queue_size=1)
        self.motion_plan_request_publisher = rospy.Publisher('/move_group/motion_plan_request',
                                                    moveit_msgs.msg.MotionPlanRequest,
                                                    queue_size=1)
        robot_state_publisher = rospy.Publisher('/robot/joint_states', JointState, queue_size=1)
        # self.start_pub = rospy.Publisher('/rviz/moveit/update_start_state', Empty, 1)
        # self.goal_pub = rospy.Publisher('/rviz/moveit/update_goal_state', Empty, 1)
        self.pos = [0 for _ in range(9)]
        self.state = JointState()
        self.state.name = [f'panda_joint{j}' for j in range(1, 8)]+['panda_finger_joint1', 'panda_finger_joint2']
        # ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
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

def traj_optimize(robot, start_cfg, target_cfg, dist_est, options):
    N_WAYPOINTS = options['N_WAYPOINTS']
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']
    MAXITER = options['MAXITER']
    history = options['history']
    init_guess = options['init_guess'] if 'init_guess' in options else None
    max_speed = options['max_speed']
    dif_weight = 1 # This should NOT be changed
    max_move_weight = options['max_move_weight']
    collision_weight = options['collision_weight']
    joint_limit_weight = options['joint_limit_weight']
    safety_margin = options['safety_margin']
    lr = options['lr']
    seed = options['seed']
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
            if init_guess is None:
                init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
            else:
                init_path = init_guess
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof)) * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        # opt = torch.optim.LBFGS([p], lr=lr)

        for step in range(MAXITER):
            def closure(all_losses=False):
                # p.data = utils.wrap2pi(p.data)
                opt.zero_grad()
                collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
                # cnt_check += len(p) # Counting collision checks
                control_points = robot.fkine(p, reuse=True)
                max_move_cost = torch.clamp((control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-max_speed**2, min=0).mean()
                # max_move_cost = torch.clamp((p[1:,:2]-p[:-1,:2]).pow(2).sum(dim=1)-1.5**2, min=0).sum() 
                joint_limit_cost = (
                    torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
                diff = (control_points[1:, -1]-control_points[:-1, -1]).pow(2).sum() # + (utils.wrap2pi(p[1:]-p[:-1])).pow(2).sum()

                constraint_loss = collision_weight * collision_score\
                    + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
                opt_loss = dif_weight * diff
                loss = constraint_loss + opt_loss
                loss.backward()
                p.grad[[0, -1]] = 0.0
                assert not torch.any(torch.isnan(p.grad))
                if all_losses:
                    return loss, (collision_score, max_move_cost, joint_limit_cost, diff, constraint_loss, opt_loss)
                return loss
            
            loss, (collision_score, max_move_cost, joint_limit_cost, diff, constraint_loss, opt_loss) = closure(all_losses=True)
            # opt.zero_grad()
            # collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            # control_points = robot.fkine(p, reuse=True)
            # max_move_cost = torch.clamp(
            #     (control_points[1:]-control_points[:-1]).pow(2).sum(dim=2)-max_speed**2, min=0).sum()
            # joint_limit_cost = (
            #     torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            # diff = (control_points[1:]-control_points[:-1]).pow(2).sum() # + (utils.wrap2pi(p[1:]-p[:-1])).pow(2).sum()

            # constraint_loss = collision_weight * collision_score \
            #     + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            # opt_loss = dif_weight * diff
            # loss = constraint_loss + opt_loss
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
            if constraint_loss <= 1e-2 or step % (MAXITER/5) == 0 or step == MAXITER-1:
            # if constraint_loss <= 1e-2 or step % 1 == 0 or step == MAXITER-1:
                print(('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, '+
                    'joint_limit={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}, COST={:.3f}').format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    joint_limit_cost.item(), joint_limit_weight,
                    diff.item(), dif_weight,
                    loss.item(), diff.sqrt().item()))
            
            opt.step(closure)
            # loss.backward()
            # p.grad[[0, -1]] = 0.0 # Do not change start and goal configuration
            # opt.step()
            # p.data = utils.wrap2pi(p.data) # Do not wrap if your configuration is not angular
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
    # This function is not recommended. 
    # You should use the single_shot function to display a trajectory
    # and use the sliding bar to animate it in rViz
    exp = DiffCoplusBaxterExperiments()
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
    exp = DiffCoplusBaxterExperiments()
    # print('Adding box')
    # rospy.sleep(2)

    # joint_names = [f'panda_joint{j}' for j in range(1, 8)]#+['panda_finger_joint1', 'panda_finger_joint2'] # panda
    joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2',
        'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

    
    # start_state = JointState()
    # start_state.header = rospy.Header()
    # start_state.header.stamp = rospy.Time.now()
    # start_state.name = joint_names[:7]
    # start_state.position = path[0].numpy().tolist() # torch.cat([path[0], torch.FloatTensor([0.035, 0.035])]).numpy().tolist()
    # start_robot_state = RobotState()
    # start_robot_state.joint_state = start_state
    # exp.move_group.set_start_state(start_robot_state)
    # # exp.go_to_joint_state(torch.cat([path[0], torch.FloatTensor([0.035, 0.035])]).numpy().tolist())
    # # exp.start_pub.publish(Empty())
    # # rospy.sleep(1)
    # target_joint_values = path[-1].numpy().tolist() # torch.cat([path[-1], torch.FloatTensor([0.035, 0.035])]).numpy().tolist() # 
    # # exp.go_to_joint_state(target_joint_values)
    # # exp.goal_pub.publish(Empty())
    # # print(target_joint_values)
    # # # rospy.sleep(1)
    # exp.move_group.set_joint_value_target(target_joint_values)
    

    # mrq = exp.move_group.construct_motion_plan_request()
    # exp.motion_plan_request_publisher.publish(mrq)
    # exp.move_group.set_planning_time(10)
    # exp.move_group.plan()
    # exp.move_group.go()

    # box_names = []
    ## Commented out because assuming obstacles already exist in scene
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
    for q in path[1:-1]:
        traj_point = JointTrajectoryPoint()
        traj_point.positions = q.numpy().tolist()# + [0.035, 0.035]
        joint_traj.points.append(traj_point)
    joint_traj.joint_names = joint_names
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory = joint_traj
    disp_traj = DisplayTrajectory()
    disp_traj.model_id = 'baxter' # 'panda'
    disp_traj.trajectory.append(robot_traj)
    disp_traj.trajectory_start.joint_state.position = path[1] # torch.cat([path[1], torch.FloatTensor([0.035, 0.035])]).numpy()
    disp_traj.trajectory_start.joint_state.name = joint_traj.joint_names
    pub.publish(disp_traj)
    rospy.sleep(5)

    print('Start:', path[0]/np.pi*180)
    print('Goal:', path[-1]/np.pi*180)
    
    return exp

def escape(robot, dist_est, start_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS']
    safety_margin = options['safety_margin']
    lr = options['lr']
    path_history = []
    init_path = start_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())

        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin
        loss = collision_score 
        loss.backward()
        opt.step()
        p.data = utils.wrap2pi(p.data)
        if collision_score <= 1e-4:
            break
    return torch.stack(path_history, dim=0)


def main():
    robot_name = 'baxter' # [baxter, panda]
    env_name = 'self' # 'bookshelvessmall' # 'catontable' #'2objontable' #'complex' # 2objontable' # 'table'
    filename = 'data/3d_baxter_self_both_arms.pt'
    if filename is None:
        dataset = torch.load('data/3d_{}_{}.pt'.format(robot_name, env_name))
    else:
        dataset = torch.load(filename)
    cfgs = dataset['data']
    cfgs = cfgs.type(torch.float)
    labels = dataset['label']
    # dists = dataset['dist']
    obstacles = dataset['obs']
    robot = dataset['robot']()
    train_num = int(len(cfgs) * 0.5)
    fkine = robot.fkine
    # '''
    #====
    # checker = DiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(1000)), beta=1.0) # kernel.FKKernel(fkine, 
    # checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]))
    # with open('results/checker_3d_{}_{}.p'.format(robot_name, env_name), 'wb') as f:
    #     pickle.dump(checker, f)
    #     print('checker saved: {}'.format(f.name))
    #==== The following can be used alone to save training time in debugging
    with open('results/checker_3d_{}_{}.p'.format(robot_name, env_name), 'rb') as f:
        checker = pickle.load(f)
        print('checker loaded: {}'.format(f.name))

    fitting_target = 'label' # {label, dist, hypo}
    Epsilon = 0.01
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine)
    dist_est = checker.rbf_score
    min_score = dist_est(cfgs[train_num:]).min().item()
    print('MIN_SCORE = {}'.format(min_score))
    safety_margin = 0 #max(1/5*min_score, -0.5)
    
    # Check DiffCo test ACC
    test_num = len(cfgs) - train_num
    test_preds = torch.zeros(test_num)
    bs = 4000
    # from tqdm import tqdm
    # def test_collision(train_num, bs, test_preds, checker: DiffCo, cfgs):
    #     for i in range(0, 4000, bs): #tqdm(
    #         test_preds[i:i+bs] = (checker.score(cfgs[train_num+i:train_num+i+bs]) > 0) * 2 - 1
    # t0 = time()
    # test_preds = (dist_est(cfgs[train_num:train_num+test_num])-safety_margin > 0) * 2 - 1
    # test_collision(train_num, bs, test_preds, checker, cfgs)
    # print(f'avg {(time()-t0)/test_num} seconds.')
    test_preds = (dist_est(cfgs[train_num:train_num+test_num])-safety_margin > 0) * 2 - 1
    test_labels = labels[train_num:train_num+test_num].reshape(test_preds.shape)
    test_acc = torch.sum(test_preds == test_labels, dtype=torch.float32)/len(test_preds)
    test_tpr = torch.sum(test_preds[test_labels==1] == 1, dtype=torch.float32) / len(test_preds[test_labels==1])
    test_tnr = torch.sum(test_preds[test_labels==-1] == -1, dtype=torch.float32) / len(test_preds[test_labels==-1])
    print('Test acc: {:.2f}, TPR {:.2f}, TNR {:.2f}'.format(test_acc, test_tpr, test_tnr))
    # assert(test_acc > 0.75)
    # return

    obs_cfgs = cfgs[labels.reshape(-1) == -1]
    indices = torch.randint(0, len(obs_cfgs), (2, ))
    while indices[0] == indices[1]:
        indices = torch.randint(0, len(obs_cfgs), (2, ))

    ## Setting start and target configurations
    # start_cfg = torch.FloatTensor([-39, 40, -111, 81, 4, 29, -136])/180*pi # free_cfgs[indices[0]]
    # start_cfg = torch.FloatTensor([-41, 27, -88, 23, -166, 67, 168])/180*pi # 2obj scene, start from between the objs
    # torch.FloatTensor([25, 31, -120, 58, -66, -8, 116])/180*pi # medium scene
    # torch.FloatTensor([5, 51, -126, 58, -66, -8, 116])/180*pi # complex scene, start below objects
    # torch.FloatTensor([-5, 49, -146, 41, -107, -20, 168])/180*pi #2obj scene, start from left side of the objects
    # start_cfg = torch.FloatTensor([-11, -1, -64, 60, 11, 36, 147])/180*pi # medium scene
    # start_cfg = torch.FloatTensor([-67, 21, 65, -41, -18, 138, 39]) / 180*pi # Panda, bookshelves small, narrow passage 1/2
    # start_cfg = torch.FloatTensor([31, 37, -32, -90, -142, 153, -154]) / 180*pi # Panda, bookshelves, narrow passage 1
    # 
    # torch.FloatTensor([-27, 34, -92, 34, -174, -50, -19]) /180*pi #start from high-risk (basic scene)
    # target_cfg = torch.FloatTensor([13, 31, -88, 16, -160, -27, 169])/180*pi # 2obj scene, stop beside table
    # torch.FloatTensor([-48, 59, -147, 50, -170, -30, 169])/180*pi #complex scene, end below objects; medium scene
    # torch.FloatTensor([-22, 28, -87, 45, -170, -30, 169])/180*pi #2obj scene, stop on the right of the objects
    # target_cfg = torch.FloatTensor([-111, -83, 95, -84, -16, 149, 137])/180*pi # Panda, bookshelves small, narrow passage 1
    # target_cfg = torch.FloatTensor([-130, -45, 92, -88, 71, 158, 14])/180*pi # Panda, bookshelves small, narrow passage 2
    # target_cfg = torch.FloatTensor([45, 44, -82, -62, 118, 161, -21]) / 180*pi # Panda, bookshelves, narrow passage 1
    # 
    # torch.FloatTensor([4, 29, -86, 44, 3, 16, -146])/180*pi
    
    # This is for trajectory optimization with or w/o an initial guess
    # with open('data/{}_{}_rrt.json'.format(robot_name, env_name), 'r') as f:
    #     init_guess = torch.FloatTensor(json.load(f)['path'])
    # start_cfg = init_guess[0]
    # target_cfg = init_guess[-1]
    if False:
        p = init_guess
        # p = cfgs[labels==1][:100]
        # print(p[62]/pi*180)
        # return
    else:
        # options = {
        #     'N_WAYPOINTS': 50,
        #     'NUM_RE_TRIALS': 1,
        #     'MAXITER': 800,
        #     'max_move_weight': 100,
        #     'collision_weight': 1,
        #     'joint_limit_weight': 100,
        #     'safety_margin': safety_margin,
        #     'max_speed': 0.10,
        #     'seed': 1079,
        #     'lr': 1e-2,
        #     'init_guess': init_guess.clone(),
        #     'history': True
        # }
        # p, path_history, num_trial, num_step = traj_optimize(robot, start_cfg, target_cfg, dist_est, options)
        # with open('results/path_3d_{}_{}_1.json'.format(robot_name, env_name), 'w') as f:
        #     json.dump(
        #         {
        #             'path': p.data.numpy().tolist(), 
        #             'path_history': [tmp.data.numpy().tolist() for tmp in path_history],
        #             'trial': num_trial,
        #             'step': num_step
        #         },
        #         f, indent=1)
        #     print('Plan recorded in {}'.format(f.name))
            # pathname = f.name

        ## This is for escaping from high-risk region
        options_escape = {
            'N_WAYPOINTS': 80,
            'safety_margin': -1.5,
            'lr': 5e-2
        }
        # start_cfg = torch.FloatTensor([65, 14, 122, -82, 0, 94, 44])/180*np.pi
        start_cfg = torch.FloatTensor([-84, -21, 8, 62, 91, 79, -132, 41, -42, 35, 126, -120, 108, 158])/180*np.pi # baxter, self collision
        p = escape(robot, dist_est, start_cfg, options_escape)
        with open('results/path_3d_{}_{}_escape.json'.format(robot_name, env_name), 'w') as f:
            json.dump(
                {
                    'path': p.data.numpy().tolist(), 
                },
                f, indent=1)
            print('Plan recorded in {}'.format(f.name))
            pathname = f.name

        ## This is for loading previous trajectories
        with open(pathname, 'r') as f:
            p = torch.FloatTensor(json.load(f)['path'])
        # with open('data/medium_success_2.json', 'r') as f:
        #     p = torch.FloatTensor(json.load(f)['path'])

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
        # box_names, exp = single_shot(torch.cat([init_guess.detach(), utils.dense_path(p, 0.1)], dim=0), obstacles)
        exp = single_shot(utils.dense_path(p, 0.1), obstacles)

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    
    # for box_name in box_names:
    #     exp.scene.remove_world_object(box_name)
    #     print('Tried removing {}!'.format(box_name))
    #     wait_for_state_update(exp.scene, box_name, box_is_attached=False, box_is_known=False)

if __name__ == "__main__":
    main()


