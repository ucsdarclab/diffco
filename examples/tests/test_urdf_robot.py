from diffco.collision_interfaces.urdf_interface import FrankaPanda, URDFRobot, robot_description_folder
import trimesh
import fcl
import numpy as np
import torch
import os
import trimesh.transformations as tf
from trimesh.collision import CollisionManager
from trimesh.primitives import Box, Sphere, Cylinder, Capsule

def test_urdf(urdf_robot: URDFRobot, show=False):
    print(urdf_robot)
    # print(list(urdf_robot.robot.link_map.values()))
    # print(urdf_robot.robot.collision_scene)
    # print(urdf_robot.collision_manager._allowed_collisions)

    num_cfgs = 1000
    cfgs = urdf_robot.rand_configs(num_cfgs)

    collision_labels = urdf_robot.collision(cfgs, other=None, show=show)
    print(f"in collision: {collision_labels.sum()}/{num_cfgs}, collision-free {(collision_labels==0).sum()}/{num_cfgs}")

    print("Verifying forward kinematics")
    fk = urdf_robot.compute_forward_kinematics_all_links(cfgs, return_collision=True)
    for i in range(num_cfgs):
        urdf_robot.robot.update_cfg(cfgs[i].numpy())
        fk = urdf_robot.compute_forward_kinematics_all_links(cfgs[i:i+1], return_collision=True)
        urdf_robot.collision_manager.set_fkine({k: [(t[0], r[0]) for t, r in v] for k, v in fk.items()})
        for link_name in fk:
            for piece_idx, cobj_name in enumerate(urdf_robot.collision_manager.link_collision_objects[link_name]):
                fk_trans, fk_rot = fk[link_name][piece_idx]
                fk_rot = fk_rot.numpy()
                fk_tf = np.eye(4, dtype=fk_rot.dtype)
                fk_tf[:3, :3] = fk_rot
                fk_tf[:3, 3] = fk_trans.numpy()
                fcl_trans = urdf_robot.collision_manager._objs[cobj_name]['obj'].getTranslation()
                fcl_rot = urdf_robot.collision_manager._objs[cobj_name]['obj'].getRotation()
                fcl_tf = np.eye(4, dtype=fcl_rot.dtype)
                fcl_tf[:3, :3] = fcl_rot
                fcl_tf[:3, 3] = fcl_trans
                t_from_urdfpy = urdf_robot.robot.get_transform(cobj_name, collision_geometry=True)
                assert np.allclose(t_from_urdfpy, fcl_tf, rtol=1e-4, atol=1e-6), \
                    f"Link {link_name} transform mismatch between fcl and urdfpy: {t_from_urdfpy - fcl_tf}"
                assert np.allclose(fk_tf, fcl_tf, rtol=1e-4, atol=1e-6), \
                    f"Link {link_name} transform mismatch between fcl and compute_forward_kinematics_all_links: {fk_tf - fcl_tf}"
    print("Forward kinematics verified")

if __name__ == "__main__":
    test_urdf(FrankaPanda(load_gripper=True, device="cpu", load_visual_meshes=True), show=False)
    test_urdf(URDFRobot(
        urdf_path=os.path.join(
            robot_description_folder, 
            "allegro/urdf/allegro_hand_description_left.urdf"),
        name="allegro",
        device="cpu",
        load_visual_meshes=True
    ), show=True)
    print("All tests passed")