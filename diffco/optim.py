# Initialized from scripts/trajectory_optim

from collections import namedtuple
import torch
import numpy as np
import time
from typing import Dict
from . import utils
from .kernel_perceptrons import DiffCo
from scipy.optimize import minimize, NonlinearConstraint


def adam_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS']  # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']  # 10
    MAXITER = options['MAXITER']  # 200
    history = options['history']

    dif_weight = 1  # This should NOT be changed
    max_move_weight = 10
    collision_weight = 10
    joint_limit_weight = 10
    safety_margin = options['safety_margin']
    max_speed = options['max_speed']
    try:
        lr = options['extra_optimizer_options']['lr']
    except KeyError:
        lr = 5e-1
    print('Adam lr = {}'.format(lr))
    seed = options['seed']
    torch.manual_seed(seed)

    lowest_loss_solution = None
    lowest_loss = np.inf
    lowest_loss_obj = np.inf
    lowest_loss_trial = None
    lowest_loss_step = None
    best_valid_solution = None
    best_valid_obj = np.inf
    best_valid_step = None
    best_valid_trial = None

    trial_histories = []
    cnt_check = 0

    def cost(p):
        # same as the other "cost" functions, just without preprocess.
        # Only for when the init_path is only two states.
        control_points = robot.fkine(p)
        obj = (control_points[1:]-control_points[:-1]).square().sum()
        return obj.data.numpy()

    found = False
    start_t = time.time()
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution'].clone()
                assert len(init_path) >= 2
                if len(init_path) == 2:
                    rec = {
                        'start_cfg': start_cfg.numpy().tolist(),
                        'target_cfg': target_cfg.numpy().tolist(),
                        'cnt_check': cnt_check,
                        'cost': cost(init_path[1:-1]).item(),
                        'time': time.time() - start_t,
                        'success': True,
                        'seed': seed,
                        'solution': init_path.numpy().tolist()
                    }
                    return rec
            else:
                init_path = torch.from_numpy(np.linspace(
                    start_cfg, target_cfg, num=N_WAYPOINTS)).double()
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * \
                (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)

        for step in range(MAXITER):
            opt.zero_grad()
            collision_score = torch.clamp(
                dist_est(p)-safety_margin, min=0).sum()
            cnt_check += len(p)  # Counting collision checks
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp(
                (control_points[1:]-control_points[:-1]).square().sum(dim=2)-max_speed**2, min=0).sum()
            joint_limit_cost = (
                torch.clamp(robot.limits[:, 0]-p, min=0) + torch.clamp(p-robot.limits[:, 1], min=0)).sum()
            diff = (control_points[1:]-control_points[:-1]).square().sum()
            constraint_loss = collision_weight * collision_score\
                + max_move_weight * max_move_cost + joint_limit_weight * joint_limit_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            # p.data = utils.wrap2pi(p.data)
            if history:
                path_history.append(p.data.clone())
            if loss.data.numpy() < lowest_loss:
                lowest_loss = loss.data.numpy()
                lowest_loss_solution = p.data.clone()
                lowest_loss_step = step
                lowest_loss_trial = trial_time
                lowest_loss_obj = objective_loss.data.numpy()
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_obj:
                    best_valid_obj = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            # if constraint_loss <= 1e-2 or step % (MAXITER/5) == 0 or step == MAXITER-1:
            #     print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
            #         trial_time, step,
            #         collision_score.item(), collision_weight,
            #         max_move_cost.item(), max_move_weight,
            #         diff.item(), dif_weight,
            #         loss.item()))
            if constraint_loss <= 1e-2 and torch.norm(p.grad) < 1e-4:
                break
        trial_histories.append(path_history)

        if best_valid_solution is not None:
            found = True
            break
    end_t = time.time()
    if not found:
        # print('Did not find a valid solution after {} trials!\
        # Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_loss_solution
        solution_step = lowest_loss_step
        solution_trial = lowest_loss_trial
        solution_obj = lowest_loss_obj
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
        solution_obj = best_valid_obj
    # Could be empty when history = false
    path_history = trial_histories[solution_trial]
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]

    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': solution_obj.item(),
        'time': end_t - start_t,
        'success': found,
        'seed': seed,
        'solution': solution.numpy().tolist()
    }
    return rec


def givengrad_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS']  # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']  # 10
    MAXITER = options['MAXITER']  # 200
    safety_margin = options['safety_margin']
    max_speed = options['max_speed']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check, obj, max_move_cost, collision_cost, joint_limit_cost, call_cnt
    global var_p_max_move, var_p_collision, var_p_limit, var_p_cost
    global latest_p_max_move, latest_p_collision, latest_p_limit, latest_p_cost
    cnt_check = 0
    call_cnt = 0

    def pre_process(p):
        global var_p
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        # p[:] = utils.wrap2pi(p)
        var_p = torch.cat([init_path[:1], p, init_path[-1:]],
                          dim=0).requires_grad_(True)
        return var_p

    def con_collision_free(p, return_tensor=False):
        global cnt_check
        if not return_tensor:
            # calling from optimizer, p = ((n-2)*dof, )
            p = pre_process(p)
        dense_p = utils.dense_path(p, max_speed)
        collision_cost = -(dist_est(dense_p[1:-1])-safety_margin)
        cnt_check += len(dense_p)
        collision_cost = torch.clamp_(collision_cost, max=0).reshape(-1)
        n_segment = len(p)-1
        n_point = len(dense_p) - 2
        multiplier = n_point // n_segment
        if n_point % n_segment != 0:
            multiplier += 1
            pad_cost = torch.zeros(n_segment*multiplier-n_point)
            collision_cost = torch.cat([collision_cost, pad_cost])
        collision_cost = collision_cost.reshape(n_segment, -1).sum(dim=1)
        return collision_cost if return_tensor else collision_cost.data.numpy()
        
    def jac_con_collision_free(p): # Full jacobian, vs only gradient of sum
        var_p_collision = pre_process(p)
        jac = torch.autograd.functional.jacobian(
            lambda x: con_collision_free(x, return_tensor=True), 
            var_p_collision,
            create_graph=False, strict=False,
            vectorize=True, strategy='reverse-mode'
        )
        n_constraints = jac.shape[0]
        return jac[:, 1:-1].numpy().reshape(n_constraints, -1)

    def con_joint_limit(p):
        global joint_limit_cost, var_p_limit, latest_p_limit
        var_p_limit = pre_process(p)
        latest_p_limit = var_p_limit.data[1:-1].numpy().reshape(-1)
        joint_limit_cost = -torch.sum(torch.clamp_(robot.limits[:, 0]-var_p_limit, min=0)
                                      + torch.clamp_(var_p_limit-robot.limits[:, 1], min=0))
        return joint_limit_cost.data.numpy()

    def grad_con_joint_limit(p):
        if all(p == latest_p_limit):
            var_p_limit.grad = None
            joint_limit_cost.backward()
            if var_p_limit.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_limit.grad[1:-1].numpy().reshape(-1)
        else:
            raise ValueError('p is not the same as the lastest passed p')

    def cost(p):
        global obj, var_p_cost, latest_p_cost
        var_p_cost = pre_process(p)
        latest_p_cost = var_p_cost.data[1:-1].numpy().reshape(-1)
        control_points = robot.fkine(var_p_cost)
        obj = (control_points[1:]-control_points[:-1]).square().sum()
        return obj.data.numpy()

    def grad_cost(p):
        if np.allclose(p, latest_p_cost):
            var_p_cost.grad = None
            obj.backward()
            if var_p_cost.grad is None:
                return np.zeros(len(p), dtype=p.dtype)
            return var_p_cost.grad[1:-1].numpy().reshape(-1)
        else:
            print(p, latest_p_cost, np.linalg.norm(p-latest_p_cost))
            raise ValueError('p is not the same as the lastest passed p')

    start_t = time.time()
    success = False
    lowest_const_loss = np.inf
    solution_rec = None
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
                assert len(init_path) >= 2
                if len(init_path) == 2:
                    rec = {
                        'start_cfg': start_cfg.numpy().tolist(),
                        'target_cfg': target_cfg.numpy().tolist(),
                        'cnt_check': cnt_check,
                        'cost': cost(init_path[1:-1]).item(),
                        'time': time.time() - start_t,
                        'success': True,
                        'seed': seed,
                        'solution': init_path.numpy().tolist()
                    }
                    return rec
            else:
                init_path = torch.from_numpy(np.linspace(
                    start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * \
                (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        print('updated slsqp')
        res = minimize(
            cost, init_path[1:-1].reshape(-1).numpy(), jac=grad_cost,
            method='slsqp',
            constraints=[
                # {'fun': con_max_move, 'type': 'ineq', 'jac': grad_con_max_move},
                {'fun': con_collision_free, 'type': 'ineq',
                'jac': jac_con_collision_free},
                {'fun': con_joint_limit, 'type': 'ineq', 'jac': grad_con_joint_limit}
            ],
            options={'maxiter': MAXITER, **options['extra_optimizer_options']}
        )
        if res.success:
            success = True
            solution_rec = res
            break
        tmp_loss = -(con_collision_free(res.x).sum() + con_joint_limit(res.x)) #10 * con_max_move(res.x) +
        if tmp_loss < lowest_const_loss:
            lowest_const_loss = tmp_loss
            solution_rec = res
    end_t = time.time()
    solution_rec.x = solution_rec.x.reshape([-1, robot.dof])
    solution_rec.x = pre_process(solution_rec.x)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': solution_rec.fun,
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': solution_rec.x.data.numpy().tolist()
    }
    return rec


def trustconstr_traj_optimize(robot, dist_est, start_cfg, target_cfg, options):
    N_WAYPOINTS = options['N_WAYPOINTS']  # 20
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']  # 10
    MAXITER = options['MAXITER']  # 200
    safety_margin = options['safety_margin']
    max_speed = options['max_speed']

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check, obj, max_move_cost, collision_cost, joint_limit_cost, call_cnt
    global var_p_max_move, var_p_collision, var_p_limit, var_p_cost
    global grad_max_move_valid, grad_collision_valid, grad_limit_valid, grad_cost_valid
    global latest_p_max_move, latest_p_collision, latest_p_limit, latest_p_cost
    cnt_check = 0
    call_cnt = 0
    grad_max_move_valid, grad_collision_valid, grad_limit_valid, grad_cost_valid = False, False, False, False

    def pre_process(p):
        global var_p
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        # p[:] = utils.wrap2pi(p)
        var_p = torch.cat([init_path[:1], p, init_path[-1:]],
                          dim=0).requires_grad_(True)
        return var_p

    def con_collision_free(p, return_tensor=False):
        global cnt_check
        if not return_tensor:
            # calling from optimizer, p = ((n-2)*dof, )
            p = pre_process(p)
        dense_p = utils.dense_path(p, max_speed)
        collision_cost = -(dist_est(dense_p[1:-1])-safety_margin)
        cnt_check += len(dense_p)
        collision_cost = torch.clamp_(collision_cost, max=0).reshape(-1)
        n_segment = len(p)-1
        n_point = len(dense_p) - 2
        multiplier = n_point // n_segment
        if n_point % n_segment != 0:
            multiplier += 1
            pad_cost = torch.zeros(n_segment*multiplier-n_point)
            collision_cost = torch.cat([collision_cost, pad_cost])
        collision_cost = collision_cost.reshape(n_segment, -1).sum(dim=1)
        return collision_cost if return_tensor else collision_cost.data.numpy()
        
    def jac_con_collision_free(p): # Full jacobian, vs only gradient of sum
        var_p_collision = pre_process(p)
        jac = torch.autograd.functional.jacobian(
            lambda x: con_collision_free(x, return_tensor=True), 
            var_p_collision,
            create_graph=False, strict=False,
            vectorize=True, strategy='reverse-mode'
        )
        n_constraints = jac.shape[0]
        return jac[:, 1:-1].numpy().reshape(n_constraints, -1)
        
    def hess_con_collision_free(p, v):
        var_p_collision = pre_process(p)
        var_v = torch.tensor(v, dtype=torch.float32)
        hess = torch.autograd.functional.hessian(
            lambda x: torch.dot(con_collision_free(x, return_tensor=True), var_v), 
            var_p_collision,
            create_graph=False, strict=False,
            vectorize=True, outer_jacobian_strategy='reverse-mode'
        )
        n_point, n_dof = var_p_collision.shape
        hess = hess[1:-1, :, 1:-1, :].numpy().reshape((n_point-2)*n_dof, -1)
        return hess

    def con_joint_limit(p):
        global joint_limit_cost, var_p_limit, latest_p_limit, grad_limit_valid
        var_p_limit = pre_process(p)
        latest_p_limit = var_p_limit.data[1:-1].numpy().reshape(-1)
        joint_limit_cost = -torch.sum(torch.clamp_(robot.limits[:, 0]-var_p_limit, min=0)
                                      + torch.clamp_(var_p_limit-robot.limits[:, 1], min=0))
        grad_limit_valid = False
        return joint_limit_cost.data.numpy()

    def grad_con_joint_limit(p):
        global grad_limit_valid
        if not all(p == latest_p_limit):
            con_joint_limit(p)
        if not grad_limit_valid:
            var_p_limit.grad = None
            joint_limit_cost.backward()
            grad_limit_valid = True
        if var_p_limit.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        grad = var_p_limit.grad[1:-1].numpy().reshape(-1)
        return grad

    def cost(p, return_tensor=False):
        global obj, var_p_cost, latest_p_cost, grad_cost_valid
        if not return_tensor:
            var_p_cost = pre_process(p)
            latest_p_cost = var_p_cost.data[1:-1].numpy().reshape(-1)
            control_points = robot.fkine(var_p_cost)
            obj = (control_points[1:]-control_points[:-1]).square().sum()
            grad_cost_valid = False
        else:
            p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0)
            control_points = robot.fkine(p)
            obj = (control_points[1:]-control_points[:-1]).square().sum()
        return obj if return_tensor else obj.data.numpy()

    def grad_cost(p):
        global grad_cost_valid
        if not np.allclose(p, latest_p_cost):
            cost(p)
        if not grad_cost_valid:
            var_p_cost.grad = None
            obj.backward()
            grad_cost_valid = True
        if var_p_cost.grad is None:
            return np.zeros(len(p), dtype=p.dtype)
        return var_p_cost.grad[1:-1].numpy().reshape(-1)
    
    def hvp_cost(p, v):
        """
         Hessian-vector product of the cost function.
         This runs slower than the BFGS update, so we don't use it.
        """
        p = torch.tensor(p, dtype=torch.float32).reshape(-1, robot.dof)
        v = torch.tensor(v, dtype=torch.float32).reshape(-1, robot.dof)
        _, hvp = torch.autograd.functional.hvp(
            lambda x: cost(x, return_tensor=True), 
            p, v,
            create_graph=False, strict=False
        )
        return hvp.numpy().reshape(-1)
    
    start_t = time.time()
    success = False
    lowest_const_loss = np.inf
    solution_rec = None
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
                assert len(init_path) >= 2
                if len(init_path) == 2:
                    rec = {
                        'start_cfg': start_cfg.numpy().tolist(),
                        'target_cfg': target_cfg.numpy().tolist(),
                        'cnt_check': cnt_check,
                        'cost': cost(init_path[1:-1]).item(),
                        'time': time.time() - start_t,
                        'success': True,
                        'seed': seed,
                        'solution': init_path.numpy().tolist()
                    }
                    return rec
            else:
                init_path = torch.from_numpy(np.linspace(
                    start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * \
                (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        res = minimize(cost, init_path[1:-1].reshape(-1).numpy(), jac=grad_cost, #hessp=hvp_cost,
                       method='trust-constr',
                       constraints=[
            NonlinearConstraint(con_collision_free, 0, np.inf, jac=jac_con_collision_free, hess=hess_con_collision_free),
            NonlinearConstraint(con_joint_limit, 0, np.inf, jac=grad_con_joint_limit)
        ],
            options={'maxiter': MAXITER, **options['extra_optimizer_options']})
        if res.success:
            success = True
            solution_rec = res
            break
        tmp_loss = -(con_collision_free(res.x).sum() + con_joint_limit(res.x))
        if tmp_loss < lowest_const_loss:
            lowest_const_loss = tmp_loss
            solution_rec = res
    end_t = time.time()

    solution_rec.x = solution_rec.x.reshape([-1, robot.dof])
    solution_rec.x = pre_process(solution_rec.x)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': solution_rec.fun,
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': solution_rec.x.data.numpy().tolist(),
        'info': solution_rec
    }
    return rec


def gradient_free_traj_optimize(robot, checker, start_cfg, target_cfg, options: Dict=None):
    N_WAYPOINTS = options['N_WAYPOINTS']
    NUM_RE_TRIALS = options['NUM_RE_TRIALS']
    MAXITER = options['MAXITER']
    max_speed = options['max_speed']
    max_dense_waypoints = options.get('max_dense_waypoints', None)

    seed = options['seed']
    torch.manual_seed(seed)

    global cnt_check
    cnt_check = 0

    def pre_process(p):
        p = torch.DoubleTensor(p).reshape([-1, robot.dof])
        # p[:] = utils.wrap2pi(p)
        p = torch.cat([init_path[:1], p, init_path[-1:]], dim=0)
        return p

    def con_collision_free(p):
        global cnt_check
        p = pre_process(p)
        dense_p = utils.dense_path(p, max_speed, max_dense_waypoints)
        collision_cost = -checker(dense_p[1:-1])
        cnt_check += len(dense_p)
        collision_cost = torch.clamp_(collision_cost, max=0).reshape(-1)
        n_segment = len(p)-1
        n_point = len(dense_p) - 2
        multiplier = n_point // n_segment
        if n_point % n_segment != 0:
            multiplier += 1
            pad_cost = torch.zeros(n_segment*multiplier-n_point)
            collision_cost = torch.cat([collision_cost, pad_cost])
        collision_cost = collision_cost.reshape(n_segment, -1).sum(dim=1)
        return collision_cost.numpy()

    def con_joint_limit(p):
        p = pre_process(p)
        return -torch.sum(torch.clamp_(robot.limits[:, 0]-p, min=0) + torch.clamp_(p-robot.limits[:, 1], min=0)).numpy()

    def cost(p):
        p_tensor = pre_process(p)
        control_points = robot.fkine(p_tensor)
        diff = (control_points[1:]-control_points[:-1]).square().sum()
        return diff.numpy()

    start_t = time.time()
    success = False
    for trial_time in range(NUM_RE_TRIALS):
        if trial_time == 0:
            if 'init_solution' in options:
                assert isinstance(options['init_solution'], torch.Tensor)
                init_path = options['init_solution']
                assert len(init_path) >= 2
                if len(init_path) == 2:
                    rec = {
                        'start_cfg': start_cfg.numpy().tolist(),
                        'target_cfg': target_cfg.numpy().tolist(),
                        'cnt_check': cnt_check,
                        'cost': cost(init_path[1:-1]).item(),
                        'time': time.time() - start_t,
                        'success': True,
                        'seed': seed,
                        'solution': init_path.numpy().tolist()
                    }
                    return rec
            else:
                init_path = torch.from_numpy(np.linspace(
                    start_cfg, target_cfg, num=N_WAYPOINTS, dtype=np.float64))
        else:
            init_path = torch.rand((N_WAYPOINTS, robot.dof)).double()
            init_path = init_path * \
                (robot.limits[:, 1]-robot.limits[:, 0]) + robot.limits[:, 0]
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        # res = minimize(
        #     cost, init_path[1:-1].reshape(-1).numpy(),
        #     method='slsqp',
        #     constraints=[
        #         # {'fun': con_max_move, 'type': 'ineq'},
        #         {'fun': con_collision_free, 'type': 'ineq'},
        #         {'fun': con_joint_limit, 'type': 'ineq'}
        #     ],
        #     options={'maxiter': MAXITER, 'iprint': 2, 'disp': True})
        
        # trust-constr seems to be more robust than slsqp
        res = minimize(cost, init_path[1:-1].reshape(-1).numpy(),
                       method='trust-constr',
                       constraints=[
            NonlinearConstraint(con_collision_free, 0, np.inf), 
            NonlinearConstraint(con_joint_limit, 0, np.inf), 
        ],
            options={'maxiter': MAXITER, **options['extra_optimizer_options']})
        if res.success:
            success = True
            break
    end_t = time.time()

    res.x = res.x.reshape([-1, robot.dof])
    res.x = pre_process(res.x)
    rec = {
        'start_cfg': start_cfg.numpy().tolist(),
        'target_cfg': target_cfg.numpy().tolist(),
        'cnt_check': cnt_check,
        'cost': res.fun,
        'time': end_t - start_t,
        'success': success,
        'seed': seed,
        'solution': res.x.numpy().tolist()
    }
    return rec


OptimizerResult = namedtuple("OptimizerResult", ["x", "misc"])
class TrajOptimizer:
    def __init__(self, robot, checker, options):
        self.robot = robot
        self.checker: DiffCo = checker
        self.options = options
        self.normalizer = lambda x: x
        self.unnormalizer = lambda x: x
        # for k in self.options:
        #     setattr(self, k, self.options[k])
        # torch.manual_seed(self.seed)

    def step(self, x):
        raise NotImplementedError

    def set_unnormalizer(self, f):
        # In step(), use unnormalizer before starting.
        self.unnormalizer = f
    
    def set_normalizer(self, f):
        # In step(), use normalizer before returning.
        self.normalizer = f

    def set_checker(self, checker):
        self.checker = checker
    
    def set_robot(self, robot):
        self.robot = robot


class Weighted(TrajOptimizer):
    def __init__(self, robot, checker, options):
        super(Weighted, self).__init__(robot, checker, options)
        self.n_waypoints = options['n_waypoints']  # 20
        # self.num_re_trials = options['num_re_trials'] # 10
        self.maxiter = options['maxiter']  # 200
        self.history = options['history']
        # options['dif_weight'] # 1 # this should not be changed
        self.dif_weight = 1
        self.max_move_weight = options['max_move_weight']  # 10
        self.collision_weight = options['collision_weight']  # 10
        self.joint_limit_weight = options['joint_limit_weight']  # 10
        self.safety_bias = options['safety_bias']
        self.max_speed = options['max_speed']
        # self.lr = options['lr'] # 5e-1
        self.optimizer = options['optimizer']
        self.optimizer_params = options['optimizer_params']
        self.dense_check = options['dense_check']
        # self.seed = options['seed']
        # torch.manual_seed(self.seed)

    def setup_logger(self, logger):
        self._logger = logger

    @torch.inference_mode(False)
    def step(self, p, maxiter=None, mask=None, write=True, verbose=False):
        assert not torch.is_inference_mode_enabled() and torch.is_grad_enabled(), \
            (f"torch inference mode = {torch.is_inference_mode_enabled()}, torch grad = {torch.is_grad_enabled()}")

        start_t = time.time()
        path_history = []
        if not isinstance(p, torch.Tensor):
            p = torch.FloatTensor(p)
        p = self.unnormalizer(p)
        p = p.to(self.checker.device)
        p.requires_grad_(True)
        assert p.requires_grad
        opt: torch.optim.Optimizer = self.optimizer(
            [p], **self.optimizer_params)
        dist_est = self.checker.rbf_score

        maxiter = maxiter if maxiter is not None else self.maxiter
        if verbose:
            self._logger.info(f'Stepping begins. Maxiter = {maxiter}')
        for step in range(maxiter):
            opt.zero_grad()
            if self.collision_weight != 0:               
                check_p = utils.dense_path(p, max_step=self.max_speed) if self.dense_check else p
                collision_score = torch.clamp(
                    dist_est(check_p)+self.safety_bias, min=0).mean() * len(p) # .sum()
            else:
                collision_score = 0
            # cnt_check += len(p) # Counting collision checks
            control_points = self.robot.fkine(p, reuse=not self.dense_check)
            if self.max_move_weight != 0:
                max_move_cost = torch.clamp(
                    (control_points[1:]-control_points[:-1]).square().sum(dim=2)-self.max_speed**2, min=0).sum()
            else:
                max_move_cost = 0
            if self.joint_limit_weight != 0:
                joint_limit_cost = (
                    torch.clamp(self.robot.limits[:, 0]-p, min=0) + torch.clamp(p-self.robot.limits[:, 1], min=0)).sum()
            else:
                joint_limit_cost = 0
            diff = (control_points[1:]-control_points[:-1]).square().sum() # .norm(dim=-1).sum() #
            constraint_loss = (
                self.collision_weight * collision_score
                + self.max_move_weight * max_move_cost
                + self.joint_limit_weight * joint_limit_cost
            )
            objective_loss = self.dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            # p.grad[[0, -1]] = 0.0
            if mask is not None:
                p.grad[~mask] = 0.0
            opt.step()
            p.data = self.robot.wrap(p.data)
            if verbose and (step % (maxiter // 5) == 0 or step+1 == maxiter):
                self._logger.info(
                    f"obj {diff:.3f}x1, "
                    f"col {collision_score:.3f}x{self.collision_weight}, "+
                    f"jnt {joint_limit_cost:.3f}x{self.joint_limit_weight}, "+
                    f"spd {max_move_cost:.3f}x{self.max_move_weight}."
                )
            if self.history:
                path_history.append(self.normalizer(p.cpu())) #data.clone()
            if constraint_loss <= 0.5: # and torch.norm(p.grad) < 1e-4:
                if verbose: 
                    self._logger.info(f"Breaking w/ Constraint={constraint_loss}.")
                break

        p = self.normalizer(p.cpu())
        end_t = time.time() -start_t
        misc = {
            'path_history': path_history,
            'time': end_t
        }
        rec = OptimizerResult(x=p, misc=misc)
        return rec