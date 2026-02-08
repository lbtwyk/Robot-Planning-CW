import pybullet as p
import pybullet_data
import numpy as np
import random
import hashlib

class DynamicAgent:
    def __init__(self, seed, bounds=[-10, 10, -10, 10]):
        self.seed = seed
        self.bounds = bounds
        self.body_id = None 
        
        # Pre-calculate trajectory parameters
        rng = random.Random(seed + 999)
        side = rng.choice(['horizontal', 'vertical'])
        if side == 'horizontal':
            self.start_node = [bounds[0]+2, 0, 0.5]
            self.end_node = [bounds[1]-2, 0, 0.5]
        else:
            self.start_node = [0, bounds[2]+2, 0.5]
            self.end_node = [0, bounds[3]-2, 0.5]
        distance = np.linalg.norm(np.array(self.end_node) - np.array(self.start_node))
        self.speed = 0.02/distance
        self.t = 0.0
        self.forward = True

    def spawn(self):
        """Creates the physical body in PyBullet."""
        if self.body_id is not None:
            return # Already spawned
            
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.0)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.0, rgbaColor=[0, 0, 1, 1])
        self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, 
                                         baseVisualShapeIndex=vis_id, basePosition=self.start_node)

    def update(self):
        if self.body_id is None: return None
        self.t += self.speed if self.forward else -self.speed
        if self.t >= 1.0 or self.t <= 0: self.forward = not self.forward
        pos = [(1-self.t)*self.start_node[i] + self.t*self.end_node[i] for i in range(2)] + [0.5]
        p.resetBasePositionAndOrientation(self.body_id, pos, [0,0,0,1])
        return pos[:2]

class RandomizedWarehouse:
    def __init__(self, seed, mode=p.GUI):
        self.seed = seed
        self.rng = random.Random(seed)
        
        if p.isConnected(): p.disconnect()
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.obstacles = []
        self.robot_id = None
        self.dynamic_agent = DynamicAgent(self.seed) 
        
        self._generate_obstacles()
        self.start_pos, self.goal_pos = self._generate_start_goal()
        self._generate_robot()

    def _generate_obstacles(self):
        num_obs = self.rng.randint(5, 8)
        for i in range(num_obs):
            pos = [self.rng.uniform(-7, 7), self.rng.uniform(-7, 7), 0.5]
            half_extents = [self.rng.uniform(0.5, 1.5), self.rng.uniform(0.5, 1.5), 0.5]
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.3, 0.3, 0.3, 1])
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, 
                                        baseVisualShapeIndex=vis_id, basePosition=pos)
            self.obstacles.append({'id': body_id, 'extents': half_extents, 'pos': pos})

    def _generate_start_goal(self):
        def is_clear(pos):
            for obs in self.obstacles:
                if (abs(pos[0] - obs['pos'][0]) < obs['extents'][0] + 1.2 and 
                    abs(pos[1] - obs['pos'][1]) < obs['extents'][1] + 1.2):
                    return False
            return True
        while True:
            s = [self.rng.uniform(-9, -7), self.rng.uniform(-9, -7)]
            g = [self.rng.uniform(7, 9), self.rng.uniform(7, 9)]
            if is_clear(s) and is_clear(g):
                return s, g
        
    def _generate_robot(self):
        self.robot_shape_type = self.rng.choice(["RECT", "POLY"])
        if self.robot_shape_type == "RECT":
            self.robot_dims = [self.rng.uniform(0.5, 1.0), self.rng.uniform(0.5, 1.0)]
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.robot_dims[0]/2, self.robot_dims[1]/2, 0.1])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.robot_dims[0]/2, self.robot_dims[1]/2, 0.1], rgbaColor=[1, 0, 0, 1])
        else:
            points = [[0.5, 0, 0], [-0.3, 0.4, 0], [-0.3, -0.4, 0]]
            indices = [0, 1, 2]
            self.robot_dims = points
            col_id = p.createCollisionShape(p.GEOM_MESH, vertices=points, indices=indices,)
            vis_id = p.createVisualShape(p.GEOM_MESH, vertices=points, indices=indices, rgbaColor=[1, 0, 0, 1])
        self.robot_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[self.start_pos[0], self.start_pos[1], 0.1])

    def activate_dynamic_obstacle(self):
        """Phase 3: Call this to add the moving agent to the scene."""
        self.dynamic_agent.spawn()

    def get_problem_setup(self):
        static_str = str([(o['pos'], o['extents']) for o in self.obstacles])
        dyn_str = str([self.dynamic_agent.start_node, self.dynamic_agent.end_node])
        combined = f"{self.seed}_{static_str}_{dyn_str}_{self.robot_dims}"
        fingerprint = hashlib.md5(combined.encode()).hexdigest()
        
        return {
            "map_bounds": [-10, 10, -10, 10],
            "start": self.start_pos,
            "goal": self.goal_pos,
            "robot_type": self.robot_shape_type,
            "robot_geometry": self.robot_dims,
            "static_obstacles": [{"pos": o['pos'][:2], "extents": o['extents'][:2]} for o in self.obstacles],
            "dynamic_obstacle": {
                "radius": 0.5,
                "path_start": self.dynamic_agent.start_node[:2],
                "path_end": self.dynamic_agent.end_node[:2],
                "speed": self.dynamic_agent.speed
            },
            "version_hash": fingerprint
        }

    def update_simulation(self):
        p.stepSimulation()
        return self.dynamic_agent.update() # Returns None if not activated