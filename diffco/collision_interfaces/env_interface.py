class PCDEnv:
    def __init__(self, point_cloud):
        raise NotImplementedError
    

class ShapeEnv:
    ''' 
    - uses a dict of shape types and params to represent environment. the dict can be updated.:
    '''
    def __init__(self, shape_dict):
        raise NotImplementedError