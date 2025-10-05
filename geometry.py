
"""
Geometry module: 2D rigid bodies (rectangular bricks) and scene generation.
"""

import numpy as np


class RigidBody:
    """Represents a 2D rectangular rigid body (brick)."""
    
    def __init__(self, x, y, width, height, angle=0.0, mass=1.0):
        """
        Parameters:
            x, y: Center position (m)
            width, height: Dimensions (m)
            angle: Rotation angle (radians, counterclockwise)
            mass: Mass (kg)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.mass = mass
        
        # Moment of inertia for rectangular body
        self.inertia = (mass / 12.0) * (width**2 + height**2)
    
    @property
    def position(self):
        return np.array([self.x, self.y])
    
    def get_vertices(self):
        """Return the 4 corner vertices in world coordinates."""
        hw, hh = self.width / 2, self.height / 2
        
        # Local coordinates (body frame)
        local_vertices = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh]
        ])
        
        # Rotation matrix
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        
        # Transform to world coordinates
        world_vertices = (R @ local_vertices.T).T + self.position
        return world_vertices
    
    def get_edges(self):
        """Return the 4 edges as pairs of vertices."""
        verts = self.get_vertices()
        edges = []
        for i in range(4):
            edges.append((verts[i], verts[(i+1) % 4]))
        return edges


def create_demo_scene(num_bricks=25, base_width=1.0, base_height=0.5):
    """
    Generate a demo scene: a pile of rectangular bricks.
    
    Creates a pyramid-like stack with some random perturbations.
    
    Parameters:
        num_bricks: Number of bricks to generate
        base_width: Brick width (m)
        base_height: Brick height (m)
    
    Returns:
        List of RigidBody objects
    """
    bodies = []
    np.random.seed(42)  # For reproducibility
    
    # Build layers
    current_y = base_height / 2  # Start just above ground
    current_layer = 0
    bricks_placed = 0
    
    while bricks_placed < num_bricks:
        # Number of bricks in this layer (decreases with height)
        layer_size = max(1, int(np.sqrt(2 * (num_bricks - bricks_placed))))
        layer_size = min(layer_size, num_bricks - bricks_placed)
        
        # Center the layer horizontally
        layer_width = layer_size * base_width
        start_x = -layer_width / 2 + base_width / 2
        
        for i in range(layer_size):
            # Position with small random offset for realism
            x = start_x + i * base_width
            x += np.random.uniform(-0.02, 0.02)
            y = current_y + np.random.uniform(-0.01, 0.01)
            
            # Small random angle perturbation
            angle = np.random.uniform(-0.05, 0.05)
            
            # Slightly varying mass
            mass = 1.0 + np.random.uniform(-0.1, 0.1)
            
            body = RigidBody(
                x=x, y=y,
                width=base_width,
                height=base_height,
                angle=angle,
                mass=mass
            )
            bodies.append(body)
            bricks_placed += 1
            
            if bricks_placed >= num_bricks:
                break
        
        # Move to next layer
        current_y += base_height * 0.95  # Slight overlap for stability
        current_layer += 1
    
    return bodies
