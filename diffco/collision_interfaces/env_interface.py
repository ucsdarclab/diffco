import trimesh
import numpy as np
import fcl


class PCDCollisionManager:
    # TODO: consider converting pcd to python-octomap
    def __init__(self, point_cloud):
        raise NotImplementedError('PCDCollisionManager is not implemented yet')
    
        self.point_cloud = point_cloud
        self._manager = fcl.DynamicAABBTreeCollisionManager()

        self.collision_object = fcl.CollisionObject(fcl.OcTree(0.1, point_cloud))
        self._manager.registerObject(self.collision_object)
    
    def update_point_cloud(self, point_cloud):
        self._manager.unregisterObject(self.collision_object)
        self.point_cloud = point_cloud
        self.collision_object = fcl.CollisionObject(fcl.OcTree(0.1, point_cloud))
        self._manager.registerObject(self.collision_object)



class PCDEnv:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.collision_manager = PCDCollisionManager(point_cloud)
    
    def update_point_cloud(self, point_cloud):
        self.collision_manager.update_point_cloud(point_cloud)
        
    

class ShapeEnv:
    ''' 
    - uses a dict of shape types and params to represent environment. the dict can be updated.:
    - the shapes are converted to a trimesh scene and further converted to a collision manager
    The format of the dict is as follows:
    shape_dict = {
        'box1': {'type': 'Box', 'params': {'extents': [1, 1, 1]}, 'transform': np.eye(4)},
        'sphere1': {'type': 'Sphere', 'params': {'radius': 1}, 'transform': np.eye(4)},
        'cylinder1': {'type': 'Cylinder', 'params': {'radius': 1, 'height': 1}, 'transform': np.eye(4)},
        'capsule1': {'type': 'Capsule', 'params': {'radius': 1, 'height': 1}, 'transform': np.eye(4)},
        'mesh1': {'type': 'Mesh', 'params': {'file_obj': 'path/to/obj'}, 'transform': np.eye(4)}
        'mesh2': {'type': 'Mesh', 'params': {'file_stl': 'path/to/stl'}, 'transform': np.eye(4)}
    }
    '''
    def __init__(self, shapes):
        self.name = 'ShapeEnv'
        self.scene = trimesh.Scene()
        for shape_name in shapes:
            shape_type = shapes[shape_name]['type']
            shape_params = shapes[shape_name]['params']
            shape_transform = shapes[shape_name].get('transform', np.eye(4))
            if shape_type == 'Box':
                self.scene.add_geometry(trimesh.primitives.Box(**shape_params), node_name=shape_name, transform=shape_transform)
            elif shape_type == 'Sphere':
                self.scene.add_geometry(trimesh.primitives.Sphere(**shape_params), node_name=shape_name, transform=shape_transform)
            elif shape_type == 'Cylinder':
                self.scene.add_geometry(trimesh.primitives.Cylinder(**shape_params), node_name=shape_name, transform=shape_transform)
            elif shape_type == 'Capsule':
                self.scene.add_geometry(trimesh.primitives.Capsule(**shape_params), node_name=shape_name, transform=shape_transform)
            elif shape_type == 'Mesh':
                if 'scale' in shape_params:
                    scale = shape_params.pop('scale')
                else:
                    scale = 1
                geom = trimesh.load(**shape_params)
                geom.apply_scale(scale)
                self.scene.add_geometry(geom, node_name=shape_name, transform=shape_transform)
        
        self.collision_manager = trimesh.collision.CollisionManager()
        self._add_scene_to_collision_manager(self.scene, self.collision_manager)
    
    def _add_scene_to_collision_manager(self, scene, collision_manager: trimesh.collision.CollisionManager):
        """
        Convert objects in trimesh.Scene to fcl CollisionObject's, keeping the names of the objects
        """
        for geometry_node_name in scene.graph.nodes_geometry:
            T, geometry = scene.graph[geometry_node_name]
            mesh = scene.geometry[geometry]
            cobj = collision_manager.add_object(name=geometry_node_name, mesh=mesh, transform=T)
        

    def remove_object(self, name):
        self.scene.delete_geometry(name)
        self.collision_manager.remove_object(name)
    
    def add_object(self, name, shape_type, shape_params, transform=np.eye(4)):
        if shape_type == 'Box':
            self.scene.add_geometry(trimesh.primitives.Box(**shape_params), node_name=name, transform=transform)
        elif shape_type == 'Sphere':
            self.scene.add_geometry(trimesh.primitives.Sphere(**shape_params), node_name=name, transform=transform)
        elif shape_type == 'Cylinder':
            self.scene.add_geometry(trimesh.primitives.Cylinder(**shape_params), node_name=name, transform=transform)
        elif shape_type == 'Capsule':
            self.scene.add_geometry(trimesh.primitives.Capsule(**shape_params), node_name=name, transform=transform)
        elif shape_type == 'Mesh':
            self.scene.add_geometry(trimesh.load(**shape_params), node_name=name, transform=transform)
        T, geometry = self.scene.graph[name]
        mesh = self.scene.geometry[geometry]
        cobj = self.collision_manager.add_object(name=name, mesh=mesh, transform=T)
    
    def update_transform(self, name, transform):
        '''
        If the transform of an object is updated, this function should be called to update the collision manager
        '''
        self.collision_manager.set_transform(name, transform)

    def update_scene(self, name=None, transform=None):
        '''
        This should only be called before visualization. It updates the scene graph with the latest transforms
        in the collision manager. It is not necessary to call this function before collision checking.
        '''
        if name is not None:
            self.scene.graph.update(frame_to=name, matrix=transform)
        else:
            for node_name in self.scene.graph.nodes_geometry:
                transform = np.eye(4)
                transform[:3, :3] = self.collision_manager._objs[node_name]['obj'].get_rotation()
                transform[:3, 3] = self.collision_manager._objs[node_name]['obj'].get_translation()
                self.scene.graph.update(frame_to=node_name, matrix=transform)