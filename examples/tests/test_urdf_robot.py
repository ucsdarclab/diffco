from diffco.collision_interfaces.urdf_interface import FrankaPanda, URDFRobot, robot_description_folder
import trimesh
import fcl
import numpy as np
import torch
import os
import trimesh.transformations as tf
from trimesh.collision import CollisionManager
from trimesh.primitives import Box, Sphere, Cylinder, Capsule

def test_urdf(urdf_robot):
    print(urdf_robot)
    print(list(urdf_robot.robot.link_map.values()))
    print(urdf_robot.robot.collision_scene)

    num_cfgs = 1000
    cfgs = torch.randn(num_cfgs, urdf_robot._n_dofs)

    collision_labels = urdf_robot.collision(cfgs, other=None, show=False)
    print(f"in collision: {collision_labels.sum()}/{num_cfgs}, collision-free {(collision_labels==0).sum()}/{num_cfgs}")

    fk = urdf_robot.compute_forward_kinematics_all_links(cfgs, return_collision=True)
    for i in range(num_cfgs):
        urdf_robot.robot.update_cfg(cfgs[i].numpy())
        for link_name, batch_pieces_pose in fk.items():
            for batch_piece_pose, cobj_name in zip(
                batch_pieces_pose, urdf_robot.collision_manager.link_collision_objects[link_name]):
                batch_piece_trans, batch_piece_rot = batch_piece_pose
                if len(batch_piece_pose) == 0 or i >= len(batch_piece_rot):
                    continue
                
                rot = batch_piece_rot[i].numpy()
                t = np.eye(4, dtype=rot.dtype)
                t[:3, :3] = rot
                t[:3, 3] = batch_piece_trans[i].numpy()
                urdf_robot.collision_manager.set_transform(cobj_name, t)
                assert np.allclose(urdf_robot.robot.get_transform(cobj_name, collision_geometry=True), t, rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    test_urdf(FrankaPanda(device="cpu"))
    test_urdf(URDFRobot(
        urdf_path=os.path.join(
            robot_description_folder, 
            "allegro/urdf/allegro_hand_description_left.urdf"),
        name="allegro",
        device="cpu"
    ))
    print("All tests passed")