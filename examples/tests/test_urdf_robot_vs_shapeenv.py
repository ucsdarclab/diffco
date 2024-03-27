from diffco.collision_interfaces.urdf_interface import FrankaPanda, URDFRobot, MultiURDFRobot, robot_description_folder
from diffco.collision_interfaces.env_interface import ShapeEnv
import trimesh
import fcl
import numpy as np
import torch
import os
import trimesh.transformations as tf
from trimesh.collision import CollisionManager
from trimesh.primitives import Box, Sphere, Cylinder, Capsule

def test_urdf_vs_shapeenv(urdf_robot: URDFRobot, shape_env: ShapeEnv, show=False):
    print(urdf_robot)
    print(shape_env)

    num_cfgs = 1000
    cfgs = urdf_robot.rand_configs(num_cfgs)

    collision_labels = urdf_robot.collision(cfgs, other=shape_env, show=show)
    print(f"in collision: {collision_labels.sum()}/{num_cfgs}, collision-free {(collision_labels==0).sum()}/{num_cfgs}")

if __name__ == "__main__":
    # Some scattered shapes with different positions and orientations
    shape_env = ShapeEnv(
        shapes={
            'box1': {'type': 'Box', 'params': {'extents': [0.1, 0.1, 0.1]}, 'transform': tf.translation_matrix([0.5, 0.5, 0.5])},
            'sphere1': {'type': 'Sphere', 'params': {'radius': 0.1}, 'transform': tf.translation_matrix([0.5, 0, 0])},
            'cylinder1': {'type': 'Cylinder', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0, -0.5, 0.5])},
            'capsule1': {'type': 'Capsule', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0.5, 0.5, 0])},
            'mesh1': {'type': 'Mesh', 'params': {'file_obj': '../../assets/object_meshes/teapot.stl', 'scale': 1e-1}, 'transform': tf.translation_matrix([0, 0.5, 0])},
        }
    )

    panda_urdf_robot = FrankaPanda(
        load_gripper=True, 
        base_transform=torch.eye(4),
        device="cpu", load_visual_meshes=True)
    test_urdf_vs_shapeenv(panda_urdf_robot, shape_env, show=True)
    
    base_transform = torch.tensor(tf.translation_matrix([-0.5, 0, 0.5]), dtype=torch.float32)
    hand_urdf_robot = URDFRobot(
        urdf_path=os.path.join(
            robot_description_folder, 
            "allegro/urdf/allegro_hand_description_left.urdf"),
        name="allegro",
        base_transform=base_transform,
        device="cpu",
        load_visual_meshes=True
    )
    test_urdf_vs_shapeenv(hand_urdf_robot, shape_env, show=False)

    multi_urdf_robot = MultiURDFRobot(
        urdf_robots=[
            panda_urdf_robot,
            hand_urdf_robot
        ],
        device="cpu"
    )
    test_urdf_vs_shapeenv(multi_urdf_robot, shape_env, show=False)

    print("All tests passed")