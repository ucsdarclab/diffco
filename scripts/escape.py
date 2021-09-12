import torch
from diffco.model import Model

class OptimSampler():
    def __init__(self, robot: Model, dist_est, args=None):
        if args is None:
            args = {}  
        self.robot = robot
        self.dist_est = dist_est
        self.N_WAYPOINTS = args['N_WAYPOINTS'] if 'N_WAYPOINTS' in args else 20
        self.safety_margin = args['safety_margin'] if 'safety_margin' in args else -0.3
        self.lr = args['lr'] if 'lr' in args else 5e-2
        self.record_freq = args['record_freq'] if 'record_freq' in args else 1
        self.post_transform = args['post_transform'] if 'post_transform' in args else None

        self.opt_args = args['opt_args'] if 'opt_args' in args else {'lr': self.lr}
        self.optimizer = args['optimizer'] if 'optimizer' in args else torch.optim.Adam

    def optim_escape(self, start_cfg):
        cfg_history = []
        init_cfg = start_cfg.clone()
        p = init_cfg.requires_grad_(True)
        opt = self.optimizer([p], **self.opt_args)

        for step in range(self.N_WAYPOINTS):
            collision_score = self.dist_est(p)-self.safety_margin
            if collision_score <= 0:
                break
            if self.record_freq and step % self.record_freq == 0:
                cfg_history.append(p.data.clone())
            opt.zero_grad()
            loss = collision_score
            loss.backward()
            opt.step()
            if self.post_transform:
                p.data = self.post_transform(p.data)
        cfg_history.append(p.data.clone())
        return torch.stack(cfg_history, dim=0), step+1 # cfg, num of diffco checks done

def resampling_escape(robot: Model, *args, **kwargs):
    rand_cfg = torch.rand(1, robot.dof)
    rand_cfg = rand_cfg * (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
    return rand_cfg