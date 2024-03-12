# Description: This file contains the interfaces for different robots and environments
# They either read robot info and provides fkine function, or read obstacle info.
# Consider add parent classes for robot and environment interfaces, respectively.

class URDFRobot:
    def __init__(self, urdf_path):
        raise NotImplementedError
    

class ROSRobot:
    def __init__(self, robot_name):
        raise NotImplementedError
    

class URDFEnv:
    def __init__(self, urdf_path):
        raise NotImplementedError
    

class PCDEnv:
    def __init__(self, point_cloud):
        raise NotImplementedError
    

class ShapeEnv:
    ''' 
    - uses a dict of shape types and params to represent environment. the dict can be updated.:
    '''
    def __init__(self, shape_dict):
        raise NotImplementedError