import unittest
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

    def test_forward_kinematics_panda(self):
        # Create an instance of the ForwardKinematicsDiffCo class
        shape_env = dc.ShapeEnv(
            shapes={
                'box1': {'type': 'Box', 'params': {'extents': [0.1, 0.1, 0.1]}, 'transform': tf.translation_matrix([0.5, 0.5, 0.5])},
                'sphere1': {'type': 'Sphere', 'params': {'radius': 0.1}, 'transform': tf.translation_matrix([0.5, 0, 0])},
                'cylinder1': {'type': 'Cylinder', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0, -0.5, 0.5])},
                'capsule1': {'type': 'Capsule', 'params': {'radius': 0.1, 'height': 0.2}, 'transform': tf.translation_matrix([0.5, 0.5, 0])},
                'mesh1': {'type': 'Mesh', 'params': {'file_obj': '../../assets/object_meshes/teapot.stl', 'scale': 1e-1}, 'transform': tf.translation_matrix([0, 0.5, 0])},
            }
        )
        panda_urdf_robot = dc.FrankaPanda(
            load_gripper=True, 
            base_transform=torch.eye(4),
            device="cpu", load_visual_meshes=False)
        fkdc = dc.ForwardKinematicsDiffCo(
            robot=panda_urdf_robot,
            environment=shape_env,
        )
        acc, tpr, tnr = fkdc.train(num_samples=3000)

        # Assert the expected result
        expected_result = 0.9 # Define the expected result here
        # self.assertGreaterEqual(acc, expected_result)
        self.assertGreaterEqual(tpr, expected_result)
        # self.assertGreaterEqual(tnr, expected_result)

    def test_forward_kinematics_two_link(self):
        shape_env = dc.ShapeEnv(
            shapes={
                'box1': {'type': 'Box', 'params': {'extents': [0.2, 0.2, 0.2]}, 'transform': tf.translation_matrix([0.5, 0.5, 0.15])},
                'sphere1': {'type': 'Sphere', 'params': {'radius': 0.1}, 'transform': tf.translation_matrix([0.5, 0, 0.15])},
            }
        )
        two_link_robot = dc.TwoLinkRobot()
        fkdc = dc.ForwardKinematicsDiffCo(
            robot=two_link_robot,
            environment=shape_env,
        )
        acc, tpr, tnr = fkdc.train(num_samples=1000)
        expected_result = 0.9
        self.assertGreaterEqual(acc, expected_result)
        self.assertGreaterEqual(tpr, expected_result)
        self.assertGreaterEqual(tnr, expected_result)

if __name__ == '__main__':
    unittest.main()