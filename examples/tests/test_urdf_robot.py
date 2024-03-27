from diffco.collision_interfaces.urdf_interface import FrankaPanda, URDFRobot, MultiURDFRobot, robot_description_folder
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
    if isinstance(urdf_robot, MultiURDFRobot):
        urdf_robots = urdf_robot.urdf_robots
        cfgs_split = urdf_robot.split_configs(cfgs)
    else:
        urdf_robots = [urdf_robot]
        cfgs_split = [cfgs]
    
    for urdf_robot, cfgs in zip(urdf_robots, cfgs_split):
        print(f"Verifying forward kinematics for {urdf_robot.name}")
        for i in range(num_cfgs):
            
            urdf_robot.urdf.update_cfg(cfgs[i].numpy())
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
                    t_from_urdfpy = urdf_robot.urdf.get_transform(cobj_name, collision_geometry=True)
                    assert np.allclose(t_from_urdfpy, fcl_tf, rtol=1e-4, atol=1e-6), \
                        f"Link {link_name} transform mismatch between fcl and urdfpy: {t_from_urdfpy - fcl_tf}"
                    assert np.allclose(fk_tf, fcl_tf, rtol=1e-4, atol=1e-6), \
                        f"Link {link_name} transform mismatch between fcl and compute_forward_kinematics_all_links: {fk_tf - fcl_tf}"
        print(f"Forward kinematics of {urdf_robot.name} verified")

if __name__ == "__main__":
    panda_urdf_robot = FrankaPanda(
        load_gripper=True, 
        base_transform=torch.eye(4),
        device="cpu", load_visual_meshes=True)
    test_urdf(panda_urdf_robot, show=False)
    
    base_transform = torch.tensor(tf.translation_matrix([0.5, 0, 0.5]), dtype=torch.float32)
    hand_urdf_robot = URDFRobot(
        urdf_path=os.path.join(
            robot_description_folder, 
            "allegro/urdf/allegro_hand_description_left.urdf"),
        name="allegro",
        base_transform=base_transform,
        device="cpu",
        load_visual_meshes=True
    )
    test_urdf(hand_urdf_robot, show=False)

    multi_urdf_robot = MultiURDFRobot(
        urdf_robots=[
            panda_urdf_robot,
            hand_urdf_robot
        ],
        device="cpu"
    )
    test_urdf(multi_urdf_robot, show=False)

    print("All tests passed")