import pickle
from diffco import DiffCo
import diffco
import torch

# import rospy
import moveit_msgs
# import moveit_commander
# import geometry_msgs.msg
# from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
# from moveit_msgs.msg import RobotState
# from tqdm import tqdm

class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'Fastronpp' in module:
            print('replaced!')
            module = module.replace('Fastronpp', 'diffco')
        return super().find_class(module, name)

if __name__ == "__main__":
    # with open('data/2d_2dof_exp1/2d_2dof_1obs_binary_00.pt', 'rb') as fp:
    #     d = torch.load(fp)
    pickle.Unpickler = RenamingUnpickler
    # d = torch.load('data/2d_2dof_exp1/2d_2dof_1obs_binary_00.pt')
    # print(d)

    from os import walk
    from os.path import join
    for root, dirs, files in walk('recovered_data'):
        for fn in sorted(files):
            if '.pt' not in fn:
                continue
            print(root, fn)
            d = torch.load(join(root, fn))
            torch.save(d, join(root, fn))
            print(root, fn)