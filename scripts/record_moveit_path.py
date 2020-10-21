import rospy
import pickle
import json
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

def record_path(msg):
    path = []
    for point in msg.trajectory[0].joint_trajectory.points:
        path.append(list(point.positions))
    with open('data/moveit_latest_path.json', 'w') as f:
        json.dump({'path': path}, f, indent=1)
        # pickle.dump(msg, f)
    print('Updated')
    print(path)

rospy.init_node('path_recorder')
path_sub = rospy.Subscriber('/move_group/display_planned_path', DisplayTrajectory, record_path)
rospy.spin()