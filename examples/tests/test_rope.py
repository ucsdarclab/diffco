import unittest
import time

import diffco as dc
import numpy as np
import torch
import trimesh.transformations as tf

class TestForwardKinematicsDiffCo(unittest.TestCase):
    def setUp(self):
        # Initialize any necessary objects or variables
        pass

    def tearDown(self):
        # Clean up any resources used in the test
        pass

    def test_forward_kinematics_rope(self):
        # Create an instance of the ForwardKinematicsDiffCo class
        print('-'*50, '\nTesting ForwardKinematicsDiffCo with Rope robot.')
        shape_env = dc.ShapeEnv(
            shapes={
                'box1': {'type': 'Box', 'params': {'extents': [0.1, 0.1, 0.1]}, 'transform': tf.translation_matrix([0.5, 0.5, 0.5])},
                'sphere1': {'type': 'Sphere', 'params': {'radius': 0.1}, 'transform': tf.translation_matrix([0.5, 0, 0])},
                'cylinder1': {'type': 'Cylinder', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0, -0.5, 0.5])},
                'capsule1': {'type': 'Capsule', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0.5, 0.5, 0])},
                'mesh1': {'type': 'Mesh', 'params': {'file_obj': '../../assets/object_meshes/teapot.stl', 'scale': 1e-1}, 'transform': tf.translation_matrix([0, 0.5, 0])},
            }
        )
        rope_urdf_robot = dc.URDFRobot(
            urdf_path='../../diffco/robot_data/rope_description/rope.urdf',
            base_transform=torch.eye(4),
            device="cpu",
            load_visual_meshes=False
        )
        fkdc = dc.ForwardKinematicsDiffCo(
            robot=rope_urdf_robot,
            environment=shape_env,
        )
        acc, tpr, tnr = fkdc.fit(num_samples=10000, verbose=True)

        # Assert the expected result
        expected_result = 0.9 # Define the expected result here
        # self.assertGreaterEqual(acc, expected_result)
        self.assertGreaterEqual(tpr, expected_result)
        # self.assertGreaterEqual(tnr, expected_result)

        # self.speed_test(fkdc)


if __name__ == '__main__':
    unittest.main()