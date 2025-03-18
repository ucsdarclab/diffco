class CollisionEnv:
    '''
    A template class for collision environment.
    Use as a reference for implementing your own collision environment.
    '''
    def __init__(self):
        pass

    def is_collision(self, qs):
        return [self._single_collision(q) for q in qs]
    
    def _single_collision(self, q):
        raise NotImplementedError

    def distance(self, qs):
        return [self._single_distance(q) for q in qs]
    
    def _single_distance(self, q):
        raise NotImplementedError
    
    def sample_q(self):
        raise NotImplementedError
    
    def plot(self, qs):
        raise NotImplementedError