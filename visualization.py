
"""
Visualization module: renders bodies and contact forces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms


def visualize_scene(bodies, contacts, output_file, force_scale=0.1):
    """
    Create a visualization of the scene with bodies and contact forces.
    
    Parameters:
        bodies: List of RigidBody objects
        contacts: List of Contact objects (with solved forces)
        output_file: Path to save the figure (PNG/SVG)
        force_scale: Scaling factor for force arrow lengths
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw ground
    x_min = min(b.x - b.width for b in bodies) - 1.0
    x_max = max(b.x + b.width for b in bodies) + 1.0
    ax.plot([x_min, x_max], [0, 0], 'k-', linewidth=2, label='Ground')
    ax.fill_between([x_min, x_max], [0, 0], [-0.2, -0.2], 
                     color='brown', alpha=0.3)
    
    # Draw bodies
    for i, body in enumerate(bodies):
        vertices = body.get_vertices()
        
        # Draw body outline
        xs = np.append(vertices[:, 0], vertices[0, 0])
        ys = np.append(vertices[:, 1], vertices[0, 1])
        ax.plot(xs, ys, 'b-', linewidth=1.5)
        ax.fill(xs, ys, color='lightblue', alpha=0.5, edgecolor='blue')
        
        # Draw center of mass
        ax.plot(body.x, body.y, 'bo', markersize=4)
        
        # Label body
        ax.text(body.x, body.y, f'{i}', fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw contact forces
    for i, contact in enumerate(contacts):
        px, py = contact.point
        
        # Draw contact point
        ax.plot(px, py, 'ro', markersize=6, zorder=10)
        
        # Draw normal force (red arrow)
        f_n = contact.f_normal
        if f_n > 1e-6:
            nx, ny = contact.normal
            ax.arrow(px, py, nx * f_n * force_scale, ny * f_n * force_scale,
                    head_width=0.05, head_length=0.05, fc='red', ec='red',
                    linewidth=2, zorder=9, alpha=0.7)
        
        # Draw friction force (green arrow)
        f_t = contact.f_tangent
        f_t_mag = np.linalg.norm(f_t)
        if f_t_mag > 1e-6:
            ax.arrow(px, py, f_t[0] * force_scale, f_t[1] * force_scale,
                    head_width=0.05, head_length=0.05, fc='green', ec='green',
                    linewidth=2, zorder=9, alpha=0.7)
        
        # Label contact with classification
        label_offset = 0.08
        label_text = f'C{i}\n{contact.classification[0].upper()}'
        ax.text(px + label_offset, py + label_offset, label_text,
               fontsize=7, ha='left', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))
    
    # Create custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow
    
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2, label='Ground'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
               markersize=8, label='Contact Point'),
        Line2D([0], [0], color='red', linewidth=2, label='Normal Force'),
        Line2D([0], [0], color='green', linewidth=2, label='Friction Force'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', 
               markersize=6, label='Body Center')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Set axis properties
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_title('2D Static Friction Contact Forces\n(S=Sticking, N=Near-cone, L=Sliding)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # Auto-scale with some padding
    y_min = min(0, min(b.y - b.height for b in bodies)) - 0.5
    y_max = max(b.y + b.height for b in bodies) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
