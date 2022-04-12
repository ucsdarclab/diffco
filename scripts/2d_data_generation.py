
import numpy as np
import torch
from diffco.model import RevolutePlanarRobot

from generate_batch_data_2d import generate_one


def main(
        env_name: str = '3d_halfnarrow',
        folder: str = 'data/landscape',
        label_type: str = 'binary', # [instance, class, binary]
        num_class: int = 2,
        dof: int = 3, # [2, 3, 7]
        num_init_points: int = 8000,
        random_seed: int = 2021,
        width: float = 0.3) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    obstacles = {
        # ('circle', (3, 2), 2), #2circle
        # ('circle', (-2, 3), 0.5), #2circle
        # ('rect', (-2, 3), (1, 1)),
        # ('rect', (1.7, 3), (2, 3)),
        # ('rect', (-1.7, 3), (2, 3)),
        # ('rect', (0, -1), (10, 1)),
        # ('rect', (8, 7), 1),
        '1rect_1circle': [('rect', (4, 3), (2, 2)),
            ('circle', (-4, -3), 1)],
        # ('rect', (4, 3), (2, 2)), # 2rect
        # ('rect', (-4, -3), (2, 2)) # 2rect
        # ('rect', (3, 2), (2, 2)) # 1rect
        '3circle': [
            ('circle', (0, 4.5), 1), #3circle
            ('circle', (-2, -3), 2), #3circle
            ('circle', (-2, 2), 1.5), #3circle
        ],
        '1rect_1circle_7d': [
            ('circle', (-2, 3), 1), #1rect_1circle_7d
            ('rect', (3, 2), (2, 2)) #1rect_1circle_7d
        ],
        '2class_1': [
            ('rect', (5, 0), (2, 2), 0), #2class_1
            ('circle', (-3, 6), 1, 1), #2class_1
            ('rect', (-5, 2), (2, 1.5), 1), #2class_1
            ('circle', (-5, -2), 1.5, 1), #2class_1 
            ('circle', (-3, -6), 1, 1) #2class_1
        ],
        '2class_2': [
            ('rect', (0, 3), (16, 0.5), 1), #2class_2
            ('rect', (0, -3), (16, 0.5), 0), #2class_2
        ],
        # ('rect', (-7, 3), (2, 2)) #1rect_active
        '3circle_7d': [
            ('circle', (-2, 2), 1), #3circle_7d
            ('circle', (-3, 3), 1), #3circle
            ('circle', (-6, -3), 1) #3circle
        ]
        # ('rect', (5, 4), (4, 4), 0), #2instance_big
        # ('circle', (-5, -4), 2, 1) #2instance_big
    }
    if env_name == '7d_narrow':
        obstacles = []
        lb = np.array([-8, 1.0], dtype=float)
        ub = np.array([8, 8], dtype=float)
        for i in range(150):
            pos = np.random.rand(2,)*(ub-lb)+lb
            pos = pos.tolist()
            size = (1, 1)
            obstacles.append(('rect', pos, size))
        
        lb = np.array([-8, -8], dtype=float)
        ub = np.array([8, -1.0], dtype=float)
        for i in range(150):
            pos = np.random.rand(2,)*(ub-lb)+lb
            pos = pos.tolist()
            size = (1, 1)
            obstacles.append(('rect', pos, size))
        link_length = 1
    elif env_name == '3d_halfnarrow':
        obstacles = []
        lb = np.array([-8, 1.0], dtype=float)
        ub = np.array([8, 8], dtype=float)
        for i in range(150):
            pos = np.random.rand(2,)*(ub-lb)+lb
            pos = pos.tolist()
            size = (1, 1)
            obstacles.append(('rect', pos, size))

        link_length = 2.5
    else:
        obstacles = obstacles[env_name]
        lengths = {2: 3.5, 3: 2, 7:1}
        link_length = lengths[dof]
    obs_num = len(obstacles)
    
    robot = RevolutePlanarRobot(link_length, width, dof) # (7, 1), (2, 3)

    generate_one(robot, folder, obs_num, obstacles, label_type, num_class, num_init_points, env_id=env_name, vis=True)
    return

if __name__ == "__main__":
    main()
