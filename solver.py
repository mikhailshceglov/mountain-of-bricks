"""
Solver module: QP formulation with soft constraints to handle infeasibility.
"""

import numpy as np
import cvxpy as cp


class ContactForceSolver:
    """
    Solves for contact forces in static equilibrium using QP with slack variables.
    
    Handles potentially infeasible configurations by relaxing equilibrium constraints.
    """
    
    def __init__(self, bodies, contacts, mu=0.5, epsilon=1e-3, gravity=9.81):
        self.bodies = bodies
        self.contacts = contacts
        self.mu = mu
        self.epsilon = epsilon
        self.gravity = gravity
        
        self.n_bodies = len(bodies)
        self.n_contacts = len(contacts)
    
    def solve(self):
        """
        Solve for contact forces with soft equilibrium constraints.
        """
        # Decision variables
        f_n = cp.Variable(self.n_contacts, nonneg=True)
        f_t = cp.Variable(self.n_contacts)  # Scalar tangential force
        
        # Slack variables for equilibrium (allows small violations)
        slack_fx = cp.Variable(self.n_bodies)
        slack_fy = cp.Variable(self.n_bodies)
        slack_m = cp.Variable(self.n_bodies)
        
        constraints = []
        
        # Build equilibrium constraints for each body
        for body_idx, body in enumerate(self.bodies):
            force_sum_x = 0.0
            force_sum_y = -body.mass * self.gravity
            moment_sum = 0.0
            
            for c_idx, contact in enumerate(self.contacts):
                if contact.body1 == body:
                    sign = -1.0
                elif contact.body2 == body:
                    sign = 1.0
                else:
                    continue
                
                # Normal force
                fn_x = sign * contact.normal[0] * f_n[c_idx]
                fn_y = sign * contact.normal[1] * f_n[c_idx]
                
                # Friction force (scalar along tangent)
                ft_x = sign * contact.tangent[0] * f_t[c_idx]
                ft_y = sign * contact.tangent[1] * f_t[c_idx]
                
                force_sum_x += fn_x + ft_x
                force_sum_y += fn_y + ft_y
                
                # Moment
                r = contact.point - body.position
                moment_arm_n = r[0] * contact.normal[1] - r[1] * contact.normal[0]
                moment_arm_t = r[0] * contact.tangent[1] - r[1] * contact.tangent[0]
                
                moment_sum += sign * (moment_arm_n * f_n[c_idx] + moment_arm_t * f_t[c_idx])
            
            # Soft equilibrium with slack
            constraints.append(force_sum_x == slack_fx[body_idx])
            constraints.append(force_sum_y == slack_fy[body_idx])
            constraints.append(moment_sum == slack_m[body_idx])
        
        # Friction cone constraints
        for c_idx in range(self.n_contacts):
            constraints.append(f_t[c_idx] <= self.mu * f_n[c_idx])
            constraints.append(f_t[c_idx] >= -self.mu * f_n[c_idx])
        
        # Objective: minimize slack (enforce equilibrium) + regularization
        penalty = 1e6  # Large penalty for equilibrium violations
        objective = cp.Minimize(
            penalty * (cp.sum_squares(slack_fx) + cp.sum_squares(slack_fy) + cp.sum_squares(slack_m)) +
            1e-6 * cp.sum_squares(f_n) +
            1e-4 * cp.sum_squares(f_t)
        )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
        except:
            try:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            except:
                problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Solver failed with status: {problem.status}")
        
        # Extract solution
        f_n_sol = f_n.value
        f_t_sol = f_t.value
        
        if f_n_sol is None or f_t_sol is None:
            raise RuntimeError("Solver returned None values")
        
        # Store results
        for c_idx, contact in enumerate(self.contacts):
            contact.f_normal = max(0, f_n_sol[c_idx])
            contact.f_tangent = f_t_sol[c_idx] * contact.tangent
            contact.v_tangent = 0.0
            self._classify_contact(contact)
    
    def _classify_contact(self, contact):
        f_tang_mag = abs(np.dot(contact.f_tangent, contact.tangent))
        f_tang_max = self.mu * contact.f_normal
        
        if f_tang_max < 1e-9:
            contact.classification = "sticking"
            contact.cone_status = "within-cone"
            return
        
        friction_ratio = f_tang_mag / f_tang_max
        
        if friction_ratio >= 0.95:
            contact.classification = "near-cone"
            contact.cone_status = "at-cone"
        elif friction_ratio >= 0.75:
            contact.classification = "near-cone"
            contact.cone_status = "within-cone"
        else:
            contact.classification = "sticking"
            contact.cone_status = "within-cone"
        
        if f_tang_mag > f_tang_max + 1e-4:
            contact.cone_status = "violation"