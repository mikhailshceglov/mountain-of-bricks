"""
Contact detection module: finds contacts between bodies and with ground.
"""

import numpy as np


class Contact:
    """Represents a contact point between two bodies or body and ground."""
    
    def __init__(self, body1, body2, point, normal, tangent, penetration=0.0):
        """
        Parameters:
            body1: First RigidBody (or None for ground)
            body2: Second RigidBody
            point: Contact point in world coordinates (2D array)
            normal: Contact normal (unit vector pointing from body1 to body2)
            tangent: Contact tangent (unit vector, perpendicular to normal)
            penetration: Penetration depth (m, positive = overlap)
        """
        self.body1 = body1  # Can be None for ground
        self.body2 = body2
        self.point = np.array(point)
        self.normal = np.array(normal)
        self.tangent = np.array(tangent)
        self.penetration = penetration
        
        # Contact forces (set by solver)
        self.f_normal = 0.0
        self.f_tangent = np.zeros(2)
        
        # Contact state (set by solver)
        self.v_tangent = 0.0  # Relative tangential velocity
        self.classification = "unknown"  # sticking/sliding/near-cone
        self.cone_status = "unknown"  # within-cone/at-cone/violation


def detect_contacts(bodies, ground_y=0.0, contact_threshold=0.05):
    """
    Detect all contacts in the scene.
    
    Simplified collision detection:
    - Body-ground contacts (vertices below ground)
    - Body-body contacts (vertex-edge and edge-edge overlaps)
    
    Parameters:
        bodies: List of RigidBody objects
        ground_y: Ground plane y-coordinate (m)
        contact_threshold: Maximum distance to consider contact (m)
    
    Returns:
        List of Contact objects
    """
    contacts = []
    
    # 1. Detect body-ground contacts
    for body in bodies:
        vertices = body.get_vertices()
        for vertex in vertices:
            if vertex[1] <= ground_y + contact_threshold:
                # Contact with ground
                point = np.array([vertex[0], ground_y])
                normal = np.array([0.0, 1.0])  # Ground normal points up
                tangent = np.array([1.0, 0.0])  # Tangent along ground
                penetration = ground_y - vertex[1]
                
                contact = Contact(
                    body1=None,  # Ground
                    body2=body,
                    point=point,
                    normal=normal,
                    tangent=tangent,
                    penetration=max(0, penetration)
                )
                contacts.append(contact)
    
    # 2. Detect body-body contacts (simplified: vertex-vertex proximity)
    for i, body1 in enumerate(bodies):
        verts1 = body1.get_vertices()
        for j, body2 in enumerate(bodies):
            if j <= i:
                continue  # Avoid duplicate pairs and self-contact
            
            verts2 = body2.get_vertices()
            
            # Check all vertex pairs for proximity
            for v1 in verts1:
                for v2 in verts2:
                    dist = np.linalg.norm(v1 - v2)
                    if dist < contact_threshold:
                        # Create contact at midpoint
                        point = (v1 + v2) / 2
                        
                        # Normal points from body1 to body2
                        if dist > 1e-10:
                            normal = (v2 - v1) / dist
                        else:
                            normal = np.array([0.0, 1.0])
                        
                        # Tangent perpendicular to normal
                        tangent = np.array([-normal[1], normal[0]])
                        
                        penetration = contact_threshold - dist
                        
                        contact = Contact(
                            body1=body1,
                            body2=body2,
                            point=point,
                            normal=normal,
                            tangent=tangent,
                            penetration=penetration
                        )
                        contacts.append(contact)
    
    # Remove duplicate contacts (same point, within tolerance)
    unique_contacts = []
    for c in contacts:
        is_duplicate = False
        for uc in unique_contacts:
            if np.linalg.norm(c.point - uc.point) < 1e-4:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_contacts.append(c)
    
    return unique_contacts
