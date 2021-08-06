from ompl import base as ob
from ompl import geometric as og
from diffco.model import Model
from diffco.utils import dense_path
import torch
from time import time

class MyObjective(ob.OptimizationObjective):
    def __init__(self, si, func):
        super(MyObjective, self).__init__(si)
        self.func = func
        self.si_ = si
        self.dof = self.si_.getStateSpace().getDimension()
    
    def motionCost(self, s1, s2):
        s1 = to_tensor(s1, self.dof)
        s2 = to_tensor(s2, self.dof)
        return ob.Cost(self.func(s1, s2))

def to_tensor(s, dof):
    return torch.tensor([s[i] for i in range(dof)], dtype=torch.double)

class ValidityCheckerWrapper:
    def __init__(self, func, dof):
        self.func = func
        self.dof = dof
        self.counter = 0
    
    def __call__(self, s):
        self.counter += 1
        if not isinstance(s, torch.Tensor):
            s = to_tensor(s, self.dof)
        return self.func(s)
    
    def reset_count(self):
        self.counter = 0
    

class MotionPlanner(object):
    def __init__(self, robot: Model, collision_checker_function, motion_cost_function):
        self.robot = robot
        self.collision_checker_function = collision_checker_function

        self.space = ob.RealVectorStateSpace(robot.dof)
        self.bounds = ob.RealVectorBounds(robot.dof)
        for i, (l, h) in enumerate(robot.limits):
            self.bounds.setLow(i, l.item())
            self.bounds.setHigh(i, h.item())
        self.space.setBounds(self.bounds)
        # self.space.distance = motion_cost_function

        self.valid_checker = ValidityCheckerWrapper(collision_checker_function, robot.dof)
        self.si = ob.SpaceInformation(self.space)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.valid_checker))

        self.pdef = ob.ProblemDefinition(self.si)
        self.objetive = MyObjective(self.si, motion_cost_function) #ob.PathLengthOptimizationObjective(self.si))
        self.pdef.setOptimizationObjective(self.objetive)
        self.planner = og.RRTstar(self.si)
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()
        self.longest_valid_segment_length = self.space.getLongestValidSegmentLength()
    
    def plan(self, start_cfg, target_cfg, args):
        start = ob.State(self.space)
        target = ob.State(self.space)
        for i, (scfg, tcfg) in enumerate(zip(start_cfg, target_cfg)):
            start[i] = float(scfg)
            target[i] = float(tcfg)
        self.pdef.clearSolutionPaths()
        self.pdef.setStartAndGoalStates(start, target)
        # self.planner.clearQuery()
        self.planner.clear()
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()
        self.valid_checker.reset_count()

        maxtime = args['maxtime']
        plan_time = time()
        solved = self.planner.solve(maxtime)
        plan_time = time() - plan_time
        path = self.pdef.getSolutionPath()

        rec = {
            'start_cfg': start_cfg.numpy().tolist(),
            'target_cfg': target_cfg.numpy().tolist(),
            'cnt_check': self.valid_checker.counter,
            'cost': None,
            'time': plan_time,
            'success': None,
            'solution': None,
        }
        if path:
            rec['success'] = True
            # rec['cost'] = path.cost(self.objetive).value()
            path = path.getStates()
            path = torch.stack([to_tensor(s, self.robot.dof) for s in path], dim=0)
            path = dense_path(path, max_step=self.longest_valid_segment_length)
            rec['cost'] = sum([self.objetive.func(path[i], path[i+1]) for i in range(len(path)-1)])
            path = path.numpy().tolist()
            rec['solution'] = path
        else:
            rec['success'] = False

        return rec

def test_custom_motion_cost():
    from diffco.model import BaxterLeftArmFK
    robot = BaxterLeftArmFK()
    def foocheck(a):
        return torch.rand(1).item() > 0.05
    def foodist(a, b):
        return float(torch.norm(a[:3]-b[:3]))
    mp = MotionPlanner(robot, foocheck, foodist)
    import torch
    path = mp.plan(torch.zeros(7)*0.0, torch.ones(7)*0.5)
    print(path)
    # print('path length = ', path.length())
    # print('path cost = ', path.cost(mp.objetive).value())
    # path = path.getStates()
    # path = [to_tensor(s, robot.dof) for s in path]
    print('check length = ', torch.sum(torch.stack([torch.norm(path[i+1]-path[i]) for i in range(len(path)-1)])))
    print('Check cost = ', torch.sum(torch.tensor([foodist(path[i+1], path[i]) for i in range(len(path)-1)])))
 
if __name__ == "__main__":
    test_custom_motion_cost()
    
    
