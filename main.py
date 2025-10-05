#!/usr/bin/env python3
"""
Main entry point for 2D static friction contact solver.
Handles CLI, scene generation, solving, and output.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from geometry import RigidBody, create_demo_scene
from contact_detection import detect_contacts
from solver import ContactForceSolver
from visualization import visualize_scene
from validation import validate_solution


def load_scene_from_json(filepath):
    """Load scene configuration from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    bodies = []
    for b in data['bodies']:
        body = RigidBody(
            x=b['x'], y=b['y'],
            width=b['width'], height=b['height'],
            angle=b.get('angle', 0.0),
            mass=b['mass']
        )
        bodies.append(body)
    
    params = {
        'gravity': data.get('gravity', 9.81),
        'mu': data.get('mu', 0.5),
        'epsilon': data.get('epsilon', 1e-3)
    }
    
    return bodies, params


def main():
    parser = argparse.ArgumentParser(
        description='2D Static Friction Contact Force Solver'
    )
    parser.add_argument('--mode', choices=['demo', 'json'], default='demo',
                       help='Execution mode: demo (auto-generated) or json (from file)')
    parser.add_argument('--num-bricks', type=int, default=25,
                       help='Number of bricks for demo mode (default: 25)')
    parser.add_argument('--input', type=str, default='scene.json',
                       help='Input JSON file for json mode')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--mu', type=float, default=0.5,
                       help='Friction coefficient (default: 0.5)')
    parser.add_argument('--epsilon', type=float, default=1e-3,
                       help='Regularization parameter (default: 1e-3)')
    parser.add_argument('--gravity', type=float, default=9.81,
                       help='Gravitational acceleration (default: 9.81)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate scene
    print(f"=== 2D Static Friction Contact Solver ===\n")
    
    if args.mode == 'demo':
        print(f"Generating demo scene with {args.num_bricks} bricks...")
        bodies = create_demo_scene(args.num_bricks)
        params = {
            'gravity': args.gravity,
            'mu': args.mu,
            'epsilon': args.epsilon
        }
    else:
        print(f"Loading scene from {args.input}...")
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found!")
            return 1
        bodies, params = load_scene_from_json(args.input)
        # Override with CLI params if provided
        if args.mu != 0.5:
            params['mu'] = args.mu
        if args.epsilon != 1e-3:
            params['epsilon'] = args.epsilon
        if args.gravity != 9.81:
            params['gravity'] = args.gravity
    
    print(f"  Bodies: {len(bodies)}")
    print(f"  Gravity: {params['gravity']} m/s²")
    print(f"  Friction coefficient μ: {params['mu']}")
    print(f"  Regularization ε: {params['epsilon']}\n")
    
    # Detect contacts
    print("Detecting contacts...")
    contacts = detect_contacts(bodies)
    print(f"  Found {len(contacts)} contacts\n")
    
    if len(contacts) == 0:
        print("Warning: No contacts detected. Check scene configuration.")
        return 1
    
    # Solve for contact forces
    print("Solving for contact forces (QP formulation)...")
    solver = ContactForceSolver(
        bodies=bodies,
        contacts=contacts,
        mu=params['mu'],
        epsilon=params['epsilon'],
        gravity=params['gravity']
    )
    
    try:
        solver.solve()
        print("  Solver converged successfully\n")
    except Exception as e:
        print(f"  Solver failed: {e}\n")
        return 1
    
    # Validate solution
    print("Validating solution...")
    validation_results = validate_solution(
        bodies=bodies,
        contacts=contacts,
        tol_force=1e-4,
        tol_moment=1e-4,
        tol_friction=1e-6
    )
    
    # Save results
    print("\nSaving results...")
    
    # Contact forces table
    contact_file = output_dir / 'contact_forces.csv'
    with open(contact_file, 'w') as f:
        f.write('contact_id,body1_id,body2_id,px,py,nx,ny,tx,ty,')
        f.write('f_normal,f_tangent_x,f_tangent_y,f_tangent_mag,')
        f.write('v_tangent,classification,cone_status\n')
        for i, c in enumerate(contacts):
            b1_id = bodies.index(c.body1) if c.body1 else -1
            b2_id = bodies.index(c.body2) if c.body2 else -1
            f.write(f'{i},{b1_id},{b2_id},')
            f.write(f'{c.point[0]:.6f},{c.point[1]:.6f},')
            f.write(f'{c.normal[0]:.6f},{c.normal[1]:.6f},')
            f.write(f'{c.tangent[0]:.6f},{c.tangent[1]:.6f},')
            f.write(f'{c.f_normal:.6f},')
            f.write(f'{c.f_tangent[0]:.6f},{c.f_tangent[1]:.6f},')
            f_tang_mag = np.linalg.norm(c.f_tangent)
            f.write(f'{f_tang_mag:.6f},')
            f.write(f'{c.v_tangent:.6f},')
            f.write(f'{c.classification},{c.cone_status}\n')
    print(f"  Contact forces: {contact_file}")
    
    # Body forces table
    body_file = output_dir / 'body_forces.csv'
    with open(body_file, 'w') as f:
        f.write('body_id,mass,x,y,total_fx,total_fy,total_moment,')
        f.write('force_residual,moment_residual\n')
        for i, body in enumerate(bodies):
            res = validation_results['bodies'][i]
            f.write(f'{i},{body.mass:.6f},{body.x:.6f},{body.y:.6f},')
            f.write(f'{res["total_force"][0]:.6f},{res["total_force"][1]:.6f},')
            f.write(f'{res["total_moment"]:.6f},')
            f.write(f'{res["force_residual"]:.6e},{res["moment_residual"]:.6e}\n')
    print(f"  Body summaries: {body_file}")
    
    # Validation report
    val_file = output_dir / 'validation.txt'
    with open(val_file, 'w') as f:
        f.write("=== VALIDATION REPORT ===\n\n")
        f.write("ACCEPTANCE CRITERIA:\n")
        criteria = validation_results['criteria']
        for key, val in criteria.items():
            status = "PASS" if val else "FAIL"
            f.write(f"  [{status}] {key}\n")
        
        f.write("\nBODY EQUILIBRIUM:\n")
        for i, res in enumerate(validation_results['bodies']):
            f.write(f"  Body {i}: ‖∑F‖ = {res['force_residual']:.6e} N, ")
            f.write(f"‖∑M‖ = {res['moment_residual']:.6e} N·m\n")
        
        f.write("\nCONTACT CLASSIFICATION:\n")
        stats = validation_results['classification_stats']
        for cls, count in stats.items():
            f.write(f"  {cls}: {count} contacts\n")
        
        f.write("\nFRICTION CONE VIOLATIONS:\n")
        violations = [v for v in validation_results['friction_violations'] if v > 1e-6]
        if violations:
            f.write(f"  Max violation: {max(violations):.6e}\n")
            f.write(f"  Number of violations: {len(violations)}\n")
        else:
            f.write("  None (all contacts satisfy friction cone)\n")
    print(f"  Validation report: {val_file}")
    
    # Visualization
    print("\nGenerating visualization...")
    viz_file = output_dir / 'visualization.png'
    visualize_scene(bodies, contacts, str(viz_file))
    print(f"  Visualization: {viz_file}")
    
    # Summary
    print("\n=== SUMMARY ===")
    all_pass = all(validation_results['criteria'].values())
    if all_pass:
        print("✓ All acceptance criteria PASSED")
    else:
        print("✗ Some acceptance criteria FAILED (see validation.txt)")
    
    print(f"\nResults saved to: {output_dir}/")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
