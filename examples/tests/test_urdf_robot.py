from diffco.collision_interfaces.urdf_interface import TwoLinkRobot, FrankaPanda, URDFRobot, MultiURDFRobot, robot_description_folder
import trimesh
import fcl
import numpy as np
import torch
import os
import trimesh.transformations as tf
from trimesh.collision import CollisionManager
from trimesh.primitives import Box, Sphere, Cylinder, Capsule

def test_urdf(urdf_robot: URDFRobot, num_cfgs=1000, show=False):
    print(urdf_robot)
    # print(list(urdf_robot.robot.link_map.values()))
    # print(urdf_robot.robot.collision_scene)

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
        print(urdf_robot.collision_manager._allowed_internal_collisions)
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
    # dual panda arms placed like human-ish
    # panda0 = FrankaPanda(
    #     load_gripper=True,
    #     base_transform=torch.eye(4, dtype=torch.float32),
    #     device="cpu", load_visual_meshes=True)
    transform1 = tf.translation_matrix([0.1, 0.0, 0.8])
    transform1[:3, :3] = tf.euler_matrix(0, np.pi/2, 0)[:3, :3]
    panda1 = FrankaPanda(
        name="panda1",
        load_gripper=True, 
        base_transform=torch.tensor(transform1, dtype=torch.float32),
        device="cpu", load_visual_meshes=True)
    transform2 = tf.translation_matrix([-0.1, 0.0, 0.8])
    transform2[:3, :3] = tf.euler_matrix(0, -np.pi/2, 0)[:3, :3]
    panda2 = FrankaPanda(
        name="panda2",
        load_gripper=True, 
        base_transform=torch.tensor(transform2, dtype=torch.float32),
        device="cpu", load_visual_meshes=True)
    multi_panda = MultiURDFRobot(
        urdf_robots=[panda1, panda2],
        device="cpu"
    )
    test_urdf(multi_panda, show=True)

    exit()

    # rope_urdf_robot = URDFRobot(
    #     urdf_path='../../diffco/robot_data/rope_description/rope.urdf',
    #     base_transform=torch.eye(4),
    #     device="cpu",
    #     load_visual_meshes=True
    # )
    # test_urdf(rope_urdf_robot, show=True)

    # exit()

    # dvrk_urdf_robot = URDFRobot(
    #     urdf_path=os.path.join(
    #         robot_description_folder, 
    #         "dvrk_model/urdf/both_sca.urdf"),
    #     name="dvrk",
    #     device="cpu",
    #     load_visual_meshes=True
    # )
    # test_urdf(dvrk_urdf_robot, show=True)

    # exit()

    two_link_robot = TwoLinkRobot()
    test_urdf(two_link_robot, show=False)

    panda_urdf_robot = FrankaPanda(
        load_gripper=True, 
        base_transform=torch.eye(4),
        device="cpu", load_visual_meshes=True)
    test_urdf(panda_urdf_robot, show=False)

    panda_simple_collision_urdf_robot = FrankaPanda(
        simple_collision=True,
        load_gripper=False,
        load_visual_meshes=False
    )
    test_urdf(panda_simple_collision_urdf_robot, show=False)
    
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

    fetch_urdf_robot = URDFRobot(
        urdf_path=os.path.join(
            robot_description_folder, 
            "fetch_description/urdf/fetch.urdf"),
        name="fetch",
        device="cpu",
        load_visual_meshes=False
    )
    fetch_urdf_robot.collision_manager._allowed_internal_collisions[('base_link', 'bellows_link2')] = 'always'
    test_urdf(fetch_urdf_robot, show=False)

    print("All tests passed")