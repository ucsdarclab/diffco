from .. import CollisionEnv

import roboticstoolbox as rtb
from roboticstoolbox.models import Panda
from spatialgeometry import Cuboid, Cylinder, Sphere, Mesh
from spatialmath import SE3
from swift import Swift

# Path: envs/rtb/panda_envs.py

class PandaEnv(CollisionEnv):
    '''
    General collision environment for Panda robot.
    Add your own objects to create a custom environment.

    Objects: dict[key: (shape type, other shape parameters[dict])]
    '''
    def __init__(self, object_info: dict=None, launch_args: dict=None):
        super().__init__()
        self.robot = Panda()
        self.robot.q = self.robot.qr
        self.env = self._launch_env(launch_args)
        self._add_objects(object_info)
    
    def _launch_env(self, launch_args: dict):
        '''
        Launch the collision environment.

        Parameters:
            launch_args: dict
        '''
        if launch_args is None:
            launch_args = {}
        env = Swift()
        env.launch(**launch_args)
        env.add(self.robot)
        return env
    
    def _add_objects(self, object_info: dict):
        '''
        Add objects to the environment.

        Parameters:
            objects: dict[shape type: shape parameters[dict]]
        '''
        self.objects = {}
        shape_class_map = {
            'box': Cuboid,
            'cuboid': Cuboid,
            'cylinder': Cylinder,
            'sphere': Sphere,
            'mesh': Mesh
        }
        for shape_key, (shape_type, shape_params) in object_info.items():
            if shape_type in shape_class_map:
                shape_class = shape_class_map[shape_type]
                shape_obj = shape_class(**shape_params)
                self.env.add(shape_obj)
                self.objects[shape_key] = shape_obj
            else:
                raise NotImplementedError
    
    def _single_collision(self, q):
        collided = [self.robot.iscollided(q, obj) for _, obj in self.objects.items()]
        return any(collided)
    
    def _single_distance(self, q):
        dists = [self.robot.closest_point(q, obj)[0] for _, obj in self.objects.items()]
        return min(dists)
    
    def sample_q(self):
        return self.robot.random_q()


class PandaSingleCylinderEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cylinder.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cylinder1': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)

class PandaThreeCylinderEnv(PandaEnv):
    '''
    Collision environment for Panda robot with three cylinders.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cylinder1': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.3, -0.5, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            }),
            'cylinder2': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            }),
            'cylinder3': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.3, 0.5, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)

class PandaSingleCuboidEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cuboid.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cuboid1': ('cuboid', {
                'scale': [0.2, 0.2, 0.2],
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)
