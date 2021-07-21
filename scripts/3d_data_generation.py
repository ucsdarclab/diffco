import sys
# sys.path.append('/home/yuheng/DiffCo/')
from diffco import DiffCo
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import BaxterLeftArmFK, PandaFK

import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from tqdm import tqdm




if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('DiffCoplusDataGenerator', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)

    # ========================== Data generqation =========================
    def wait_for_state_update(box_name, box_is_known=False, box_is_attached=False, timeout=4):
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
    
    obstacles = [
        # ('circle', (3, 2), 2),
        # ('circle', (-2, 3), 1),
        # ('rect', (-2, 3), (1, 1)),
        # ('rect', (1, 0, 0), (0.3, 0.3, 0.3)),
        # ('rect', (-1.7, 3), (2, 3)),
        # ('rect', (0, -1), (10, 1)),
        # ('rect', (8, 7), 1),
        ]
    box_names = []
    rospy.sleep(2)
    # for i, obs in enumerate(obstacles):
    #     box_name = 'box_{}'.format(i)
    #     box_names.append(box_name)
    #     box_pose = geometry_msgs.msg.PoseStamped()
    #     # box_pose.header.frame_id = "base"
    #     box_pose.header.frame_id = robot.get_planning_frame()
    #     # box_pose.pose.orientation.w = 1.0
    #     box_pose.pose.position.x = obs[1][0]
    #     box_pose.pose.position.y = obs[1][1]
    #     box_pose.pose.position.z = obs[1][2]
    #     scene.add_box(box_name, box_pose, size=obs[2])
    #     wait_for_state_update(box_name, box_is_known=True)
    
    
    env_name = 'bookshelvessmall'

    robot_name = 'panda'
    DOF = 7
    robot = PandaFK()

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
    cfgs = torch.rand((num_init_points, DOF), dtype=torch.float32)
    cfgs = cfgs * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
    labels = torch.zeros(num_init_points, dtype=torch.float)
    # dists = torch.zeros(num_init_points, dtype=torch.float)
    # req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    
    rs = RobotState()
    # rs.joint_state.name = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'] # Baxter
    rs.joint_state.name = [f'panda_joint{j}' for j in range(1, 8)] # panda
    gsvr = GetStateValidityRequest()
    gsvr.robot_state = rs
    gsvr.group_name = group_name
    for i, cfg in enumerate(tqdm(cfgs)):
        rs.joint_state.position = cfg
        result = sv_srv.call(gsvr)

        in_collision = not result.valid
        labels[i] = 1 if in_collision else -1
        # if in_collision:
        #     depths = torch.tensor([c.penetration_depth for c in rdata.result.contacts])
        #     dists[i] = depths.abs().max()
        # else:
        #     ddata = fcl.DistanceData()
        #     robot_manager.distance(obs_manager, ddata, fcl.defaultDistanceCallback)
        #     dists[i] = -ddata.result.min_distance
    print('{} collisons, {} free'.format(torch.sum(labels==1), torch.sum(labels==-1)))
    dataset = {'data': cfgs, 'label': labels, 'obs': obstacles, 'robot': robot.__class__}
    torch.save(dataset, '/home/yuheng/DiffCo/data/3d_{}_{}.pt'.format(robot_name, env_name))
    # input()
    # for box_name in box_names:
    #     scene.remove_world_object(box_name)
    #     wait_for_state_update(box_name, box_is_attached=False, box_is_known=False)