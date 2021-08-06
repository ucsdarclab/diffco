import torch
from diffco.model import Model

def optim_escape(robot: Model, dist_est, start_cfg, args=None):
    if args is None:
        args = {}
    N_WAYPOINTS = args['N_WAYPOINTS'] if 'N_WAYPOINTS' in args else 20
    safety_margin = args['safety_margin'] if 'safety_margin' in args else -0.3
    lr = args['lr'] if 'lr' in args else 5e-2
    record_freq = args['record_freq'] if 'record_freq' in args else 1
    post_transform = args['post_transform'] if 'post_transform' in args else None
    cfg_history = []
    init_cfg = start_cfg.clone()
    p = init_cfg.requires_grad_(True)

    opt_args = args['opt_args'] if 'opt_args' in args else {'lr': lr}
    opt = args['optimizer']([p], **opt_args) if 'optimizer' in args else torch.optim.Adam([p], lr=lr)

    for step in range(N_WAYPOINTS):
        collision_score = dist_est(p)-safety_margin
        if collision_score <= 1e-4:
            break
        if record_freq and step % record_freq == 0:
            cfg_history.append(p.data.clone())
        opt.zero_grad()
        loss = collision_score
        loss.backward()
        opt.step()
        if post_transform:
            p.data = post_transform(p.data)
    cfg_history.append(p.data.clone())
    return torch.stack(cfg_history, dim=0)

def resampling_escape(robot: Model, *args, **kwargs):
    rand_cfg = torch.rand(1, robot.dof)
    rand_cfg = rand_cfg * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
    return rand_cfg