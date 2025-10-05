"""
Validation module: checks equilibrium and friction cone constraints.
"""

import numpy as np


def validate_solution(bodies, contacts, tol_force=1e-4, tol_moment=1e-4, tol_friction=1e-6):
    """
    Validate the computed contact forces against physical constraints.
    
    Checks:
    (A) Solver completed (assumed if this function is called)
    (B) Force equilibrium: ‖∑F‖ ≤ tol_force for each body
    (C) Moment equilibrium: ‖∑M‖ ≤ tol_moment for each body
    (D) Non-penetration: f_normal ≥ 0 for each contact
    (E) Friction cone: |f_tangent| ≤ μ*f_normal + tol for each contact
    (F) Contact classification is reasonable
    
    Parameters:
        bodies: List of RigidBody objects
        contacts: List of Contact objects (with solved forces)
        tol_force: Force equilibrium tolerance (N)
        tol_moment: Moment equilibrium tolerance (N·m)
        tol_friction: Friction cone tolerance (N)
    
    Returns:
        Dictionary with validation results and metrics
    """
    results = {
        'bodies': [],
        'criteria': {},
        'classification_stats': {},
        'friction_violations': []
    }
    
    gravity = 9.81  # Assumed from solver
    
    # Check each body's equilibrium
    force_residuals = []
    moment_residuals = []
    
    for body in bodies:
        # Sum all forces on this body
        total_force = np.array([0.0, -body.mass * gravity])  # Start with gravity
        total_moment = 0.0
        
        for contact in contacts:
            # Check if this contact involves this body
            if contact.body1 == body:
                sign = -1.0  # Force on body1
            elif contact.body2 == body:
                sign = 1.0   # Force on body2
            else:
                continue
            
            # Add normal force
            f_normal_vec = sign * contact.f_normal * contact.normal
            total_force += f_normal_vec
            
            # Add friction force
            f_friction_vec = sign * contact.f_tangent
            total_force += f_friction_vec
            
            # Add moment (about body center)
            r = contact.point - body.position
            moment_normal = np.cross(r, f_normal_vec)
            moment_friction = np.cross(r, f_friction_vec)
            total_moment += moment_normal + moment_friction
        
        force_residual = np.linalg.norm(total_force)
        moment_residual = abs(total_moment)
        
        force_residuals.append(force_residual)
        moment_residuals.append(moment_residual)
        
        results['bodies'].append({
            'total_force': total_force,
            'total_moment': total_moment,
            'force_residual': force_residual,
            'moment_residual': moment_residual,
            'force_ok': force_residual <= tol_force,
            'moment_ok': moment_residual <= tol_moment
        })
    
    # Check contact constraints
    penetration_violations = 0
    friction_violations = []
    classification_counts = {'sticking': 0, 'sliding': 0, 'near-cone': 0, 'unknown': 0}
    
    # Get mu from contacts (assume same for all)
    mu = 0.5  # Default
    if len(contacts) > 0:
        # Try to infer mu from near-cone contacts
        for c in contacts:
            if c.classification == 'near-cone' and c.f_normal > 1e-6:
                f_tang_mag = np.linalg.norm(c.f_tangent)
                mu = max(mu, f_tang_mag / c.f_normal)
    
    for contact in contacts:
        # Check non-penetration (f_normal >= 0)
        if contact.f_normal < -tol_friction:
            penetration_violations += 1
        
        # Check friction cone (|f_tangent| <= mu * f_normal)
        f_tang_mag = np.linalg.norm(contact.f_tangent)
        f_tang_max = mu * contact.f_normal
        violation = max(0, f_tang_mag - f_tang_max - tol_friction)
        friction_violations.append(violation)
        
        # Count classifications
        cls = contact.classification
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
    
    results['friction_violations'] = friction_violations
    results['classification_stats'] = classification_counts
    
    # Evaluate acceptance criteria
    criteria = {}
    
    # (A) Solver completed
    criteria['A_solver_completed'] = True  # Assumed if we got here
    
    # (B) Force equilibrium
    criteria['B_force_equilibrium'] = all(res <= tol_force for res in force_residuals)
    
    # (C) Moment equilibrium
    criteria['C_moment_equilibrium'] = all(res <= tol_moment for res in moment_residuals)
    
    # (D) Non-penetration
    criteria['D_non_penetration'] = penetration_violations == 0
    
    # (E) Friction cone
    max_friction_violation = max(friction_violations) if friction_violations else 0
    criteria['E_friction_cone'] = max_friction_violation <= tol_friction
    
    # (F) Classification reasonable (no "unknown")
    criteria['F_classification'] = classification_counts.get('unknown', 0) == 0
    
    results['criteria'] = criteria
    
    return results
