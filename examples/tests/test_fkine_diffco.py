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

    def test_forward_kinematics_panda(self):
        # Create an instance of the ForwardKinematicsDiffCo class
        print('-'*50, '\nTesting ForwardKinematicsDiffCo with Panda robot.')
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
            simple_collision=False,
            load_gripper=False, 
            base_transform=torch.eye(4),
            device="cpu", load_visual_meshes=False)
        fkdc = dc.ForwardKinematicsDiffCo(
            robot=panda_urdf_robot,
            environment=shape_env,
        )
        acc, tpr, tnr = fkdc.fit(num_samples=3000, verbose=True)

        # Assert the expected result
        expected_result = 0.9 # Define the expected result here
        # self.assertGreaterEqual(acc, expected_result)
        self.assertGreaterEqual(tpr, expected_result)
        # self.assertGreaterEqual(tnr, expected_result)

        self.speed_test(fkdc)

    def test_forward_kinematics_two_link(self):
        print('-'*50, '\nTesting ForwardKinematicsDiffCo with TwoLink robot.')
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
        acc, tpr, tnr = fkdc.fit(num_samples=1000, verbose=True)
        expected_result = 0.88
        self.assertGreaterEqual(acc, expected_result)
        self.assertGreaterEqual(tpr, expected_result)
        self.assertGreaterEqual(tnr, expected_result)

        self.speed_test(fkdc)


    def speed_test(self, checker):
        # Compare the speed of DiffCo with the gt collision function
        num_configs = 1000
        q = checker.robot.rand_configs(num_configs)

        dc_start_time = time.time()
        dc_collisions = checker.collision(q)
        dc_time = time.time() - dc_start_time
        gt_start_time = time.time()
        gt_collisions = checker.gt_check_func(q)
        gt_time = time.time() - gt_start_time
        
        print(f'GT collision check took {gt_time:.4f} seconds on {num_configs} configs.')
        print(f'DiffCo collision check took {dc_time:.4f} seconds on {num_configs} configs.')
        # self.assertLess(dc_time, gt_time / 20)

        # Compare the speed on a single configuration
        num_configs = 10

        dc_start_time = time.time()
        dc_collisions = checker.collision(q[:num_configs])
        dc_time = time.time() - dc_start_time
        gt_start_time = time.time()
        gt_collisions = checker.gt_check_func(q[:num_configs])
        gt_time = time.time() - gt_start_time
        
        print(f'GT collision check took {gt_time:.4f} seconds on {num_configs} configs.')
        print(f'DiffCo collision check took {dc_time:.4f} seconds on {num_configs} configs.')
        # self.assertLessEqual(dc_time, gt_time+1e-4)

    
    def test_active_learning_twolink(self):
        print('-'*50, '\nTesting active learning of ForwardKinematicsDiffCo with TwoLink robot.')
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
        acc, tpr, tnr = fkdc.fit(num_samples=1000, verbose=True)

        shape_env.update_transform('box1', tf.translation_matrix([0.4, 0.4, 0.15]))
        shape_env.update_transform('sphere1', tf.translation_matrix([0.4, -0.1, 0.15]))
        print('Shape env updated.\n'+ '-'*50)
        print('Verifying before DiffCo update:')
        acc, tpr, tnr = fkdc.verify(verbose=True)
        self.assertLess(tpr, 0.9)
        print('-'*50)

        update_start_time = time.time()
        fkdc.update(num_samples=200)
        print(f'Verifying after DiffCo update that took {time.time() - update_start_time:.4f} seconds.')
        acc, tpr, tnr = fkdc.verify(verbose=True)
        self.assertGreaterEqual(tpr, 0.9)
        print('-'*50)

    def test_active_learning_panda(self):
        print('-'*50, '\nTesting active learning of ForwardKinematicsDiffCo with Panda robot.')
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
            simple_collision=False,
            load_gripper=False, 
            base_transform=torch.eye(4),
            device="cpu", load_visual_meshes=False)
        fkdc = dc.ForwardKinematicsDiffCo(
            robot=panda_urdf_robot,
            environment=shape_env,
        )
        acc, tpr, tnr = fkdc.fit(num_samples=3000)

        shape_env.update_transform('box1', tf.translation_matrix([0.4, 0.4, 0.4]))
        shape_env.update_transform('sphere1', tf.translation_matrix([0.4, 0, 0.4]))
        shape_env.update_transform('cylinder1', tf.translation_matrix([0, -0.4, 0.4]))
        shape_env.update_transform('capsule1', tf.translation_matrix([0.4, 0.4, 0]))
        shape_env.update_transform('mesh1', tf.translation_matrix([0, 0.4, 0]))
        print('Shape env updated.\n'+ '-'*50)
        print('Verifying before DiffCo update:')
        acc, tpr, tnr = fkdc.verify(verbose=True)
        self.assertLess(tpr, 0.9)
        print('-'*50)

        update_start_time = time.time()
        fkdc.update(num_samples=200)
        print(f'Verifying after DiffCo update that took {time.time() - update_start_time:.4f} seconds.')
        acc, tpr, tnr = fkdc.verify(verbose=True)
        self.assertGreaterEqual(tpr, 0.9)
        print('-'*50)


if __name__ == '__main__':
    unittest.main()