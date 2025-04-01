import numpy as np


def elastic_collision(m1, v1, I1, omega1, r1,
                      m2, v2, I2, omega2, r2,
                      rc, n, e=1.0):
    """
    Compute the post-collision velocities for two rigid bodies undergoing an elastic collision.

    Parameters:
        m1, m2 : float
            Masses of body 1 and body 2.
        v1, v2 : ndarray (3,)
            Initial linear velocities.
        I1, I2 : ndarray (3,3)
            Moment of inertia tensors.
        omega1, omega2 : ndarray (3,)
            Initial angular velocities.
        r1, r2 : ndarray (3,)
            Center of mass positions.
        rc : ndarray (3,)
            Contact point.
        n : ndarray (3,)
            Collision normal (unit vector from body 1 to body 2).
        e : float
            Coefficient of restitution (1 for elastic collision).

    Returns:
        v1_new, omega1_new : ndarray (3,)
            Updated velocity and angular velocity of body 1.
        v2_new, omega2_new : ndarray (3,)
            Updated velocity and angular velocity of body 2.
    """
    # Compute relative velocity at contact point
    r1c = rc - r1  # Vector from CoM of body 1 to contact point
    r2c = rc - r2  # Vector from CoM of body 2 to contact point

    v_c1 = v1 + np.cross(omega1, r1c)
    v_c2 = v2 + np.cross(omega2, r2c)

    v_rel = v_c1 - v_c2  # Relative velocity
    v_rel_n = np.dot(v_rel, n) * n  # Normal component
    v_rel_t = v_rel - v_rel_n  # Tangential component

    # If objects are separating, no impulse is applied
    if np.dot(v_rel, n) >= 0:
        return v1, omega1, v2, omega2

    # Compute impulse magnitude J
    I1_inv = np.linalg.inv(I1)  # Inverse inertia tensor
    I2_inv = np.linalg.inv(I2)

    term1 = 1 / m1 + 1 / m2
    term2 = np.dot(n, np.cross(I1_inv @ np.cross(r1c, n), r1c))
    term3 = np.dot(n, np.cross(I2_inv @ np.cross(r2c, n), r2c))

    J = -(1 + e) * np.dot(v_rel, n) / (term1 + term2 + term3)

    # Compute impulse vector
    J_vec = J * n

    # Update linear velocities
    v1_new = v1 + J_vec / m1
    v2_new = v2 - J_vec / m2

    # Update angular velocities
    omega1_new = omega1 + I1_inv @ np.cross(r1c, J_vec)
    omega2_new = omega2 - I2_inv @ np.cross(r2c, J_vec)

    return v1_new, omega1_new, v2_new, omega2_new


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_sphere(ax, center, radius, color='b'):
    """Plot a sphere representing the ball."""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.5)


def plot_cylinder(ax, center, radius, height, color='r'):
    """Plot a vertical cylinder centered at `center`."""
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(-height / 2, height / 2, 10)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = z + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.5)


def find_collision_normal(ball_center, ball_radius, cyl_center, cyl_radius):
    """Compute the collision normal (unit vector from the cylinder surface to the ball)."""
    # Project ball center onto cylinder's radial plane
    closest_x = cyl_center[0] + cyl_radius * (ball_center[0] - cyl_center[0]) / np.linalg.norm(
        ball_center[:2] - cyl_center[:2])
    closest_y = cyl_center[1] + cyl_radius * (ball_center[1] - cyl_center[1]) / np.linalg.norm(
        ball_center[:2] - cyl_center[:2])
    closest_z = ball_center[2]  # Assuming cylinder is vertical
    contact_point = np.array([closest_x, closest_y, closest_z])

    # Compute normal
    normal = (ball_center - contact_point) / np.linalg.norm(ball_center - contact_point)
    return normal, contact_point


# Example usage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ball and cylinder parameters
ball_center = np.array([2.0, 1.0, 0.5])
ball_radius = 0.5
cyl_center = np.array([0.0, 0.0, 0.0])
cyl_radius = 1.0
cyl_height = 3.0

# Compute collision normal
n, contact = find_collision_normal(ball_center, ball_radius, cyl_center, cyl_radius)

# Plot objects
plot_sphere(ax, ball_center, ball_radius)
plot_cylinder(ax, cyl_center, cyl_radius, cyl_height)
ax.quiver(contact[0], contact[1], contact[2], n[0], n[1], n[2], color='g', length=0.5, linewidth=2)

# Labels and view settings
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ball-Cylinder Collision')
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
ax.set_zlim(-2, 2)
plt.show()
