"""
Solver module: QP formulation and solution for contact forces with friction.

Implements Baraff's §8.1 ε-regularization approach for static friction.
"""

import numpy as np
import cvxpy as cp


class ContactForceSolver:
    """
    Solves for contact forces in static equilibrium using QP.
    
    Uses ε-approximation for static friction:
        f_tangent = -μ * f_normal * v_tangent / max(|v_tangent|, ε)
    
    This smoothly interpolates from 0 to μ*f_normal as |v_tangent| goes from 0 to ε.
    """
    
    def __init__(self, bodies, contacts, mu=0.5, epsilon=1e-3, gravity=9.81):
        """
        Parameters:
            bodies: List of RigidBody objects
            contacts: List of Contact objects
            mu: Coulomb friction coefficient
            epsilon: Regularization parameter (m/s)
            gravity: Gravitational acceleration (m/s²)
        """
        self.bodies = bodies
        self.contacts = contacts
        self.mu = mu
        self.epsilon = epsilon
        self.gravity = gravity
        
        self.n_bodies = len(bodies)
        self.n_contacts = len(contacts)
    
    def solve(self):
        """
        Solve the QP for contact forces.
        
        Decision variables:
            - f_n[i]: Normal force magnitude at contact i
            - v_t[i]: Tangential velocity at contact i (unknown in static case)
        
        The ε-regularization allows us to treat friction as a smooth function.
        """
        # Decision variables
        f_n = cp.Variable(self.n_contacts, nonneg=True)  # Normal forces
        v_t = cp.Variable(self.n_contacts)  # Tangential velocities
        
        # For static case, we assume very small tangential velocities
        # The ε-regularization will handle the transition to static friction
        
        # Build equilibrium constraints for each body
        constraints = []
        
        for body_idx, body in enumerate(self.bodies):
            # Sum of forces on this body
            force_sum_x = 0.0
            force_sum_y = -body.mass * self.gravity  # Gravity
            moment_sum = 0.0
            
            for c_idx, contact in enumerate(self.contacts):
                # Check if this contact involves this body
                if contact.body1 == body:
                    sign = -1.0  # Force on body1
                elif contact.body2 == body:
                    sign = 1.0   # Force on body2
                else:
                    continue
                
                # Normal force contribution
                fn_x = sign * contact.normal[0] * f_n[c_idx]
                fn_y = sign * contact.normal[1] * f_n[c_idx]
                
                # Friction force contribution (ε-regularized)
                # f_tangent = -μ * f_n * v_t / max(|v_t|, ε)
                # Approximate: f_tangent ≈ -μ * f_n * v_t / ε for small v_t
                ft_x = -sign * contact.tangent[0] * self.mu * f_n[c_idx] * v_t[c_idx] / self.epsilon
                ft_y = -sign * contact.tangent[1] * self.mu * f_n[c_idx] * v_t[c_idx] / self.epsilon
                
                force_sum_x += fn_x + ft_x
                force_sum_y += fn_y + ft_y
                
                # Moment contribution (about body center)
                r = contact.point - body.position
                moment_arm_normal = r[0] * contact.normal[1] - r[1] * contact.normal[0]
                moment_arm_tangent = r[0] * contact.tangent[1] - r[1] * contact.tangent[0]
                
                moment_sum += sign * moment_arm_normal * f_n[c_idx]
                moment_sum += -sign * moment_arm_tangent * self.mu * f_n[c_idx] * v_t[c_idx] / self.epsilon
            
            # Equilibrium constraints (forces and moments sum to zero)
            constraints.append(force_sum_x == 0)
            constraints.append(force_sum_y == 0)
            constraints.append(moment_sum == 0)
        
        # Additional constraint: limit tangential velocities to reasonable range
        for c_idx in range(self.n_contacts):
            constraints.append(cp.abs(v_t[c_idx]) <= 10 * self.epsilon)
        
        # Objective: minimize kinetic energy proxy (minimize velocities)
        # This encourages static solution
        objective = cp.Minimize(
            cp.sum_squares(f_n) * 1e-6 +  # Small regularization for normal forces
            cp.sum_squares(v_t)             # Minimize tangential velocities
        )
        
        # Solve QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Solver failed with status: {problem.status}")
        
        # Extract solution
        f_n_sol = f_n.value
        v_t_sol = v_t.value
        
        # Store results in contact objects
        for c_idx, contact in enumerate(self.contacts):
            contact.f_normal = f_n_sol[c_idx]
            
            # Friction force magnitude (ε-regularized)
            v_tang = v_t_sol[c_idx]
            f_tang_mag = self.mu * f_n_sol[c_idx] * min(abs(v_tang), self.epsilon) / self.epsilon
            
            # Friction force direction (opposes tangential velocity)
            if abs(v_tang) > 1e-10:
                f_tang_dir = -np.sign(v_tang) * contact.tangent
            else:
                f_tang_dir = np.zeros(2)
            
            contact.f_tangent = f_tang_mag * f_tang_dir
            contact.v_tangent = v_tang
            
            # Classify contact
            self._classify_contact(contact)
    
    def _classify_contact(self, contact):
        """
        Classify contact as sticking/sliding/near-cone.
        
        Classification rules:
        - sticking: |v_tangent| < ε and |f_tangent| < μ * f_normal
        - sliding: |v_tangent| >= ε (saturated friction)
        - near-cone: |f_tangent| ≈ μ * f_normal (at friction limit)
        """
        f_tang_mag = np.linalg.norm(contact.f_tangent)
        f_tang_max = self.mu * contact.f_normal
        
        v_tang_abs = abs(contact.v_tangent)
        
        # Classification
        if v_tang_abs >= self.epsilon * 0.9:
            contact.classification = "sliding"
        elif f_tang_mag >= f_tang_max * 0.95:
            contact.classification = "near-cone"
        else:
            contact.classification = "sticking"
        
        # Friction cone status
        cone_violation = f_tang_mag - f_tang_max
        if cone_violation > 1e-6:
            contact.cone_status = "violation"
        elif abs(cone_violation) <= 1e-6:
            contact.cone_status = "at-cone"
        else:
            contact.cone_status = "within-cone"
