import torch

class RobotInterfaceBase:
    def __init__(self, name='', device='cpu'):
        self.name = name
        self._device = torch.device(device)
        self.base_transform = None
        self.joint_limits = None
        self._n_dof = None
        self._controlled_joints = None
        self._mimic_joints = None
        self._bodies = None
        self._body_name_to_idx_map = None
    
    def rand_configs(self, num_configs):
        raise NotImplementedError

    def collision(self, q):
        '''
        Args:
            q: joint angles [batch_size x n_dofs]
        Returns a boolean tensor of size (batch_size,) indicating whether each configuration is in collision
        '''
        raise NotImplementedError
    
    def compute_forward_kinematics_all_links(self, q, return_collision=False):
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link
            return_collision: whether to return collision geometry transforms

        Returns: translation and rotation of the link frame

        """
        raise NotImplementedError
