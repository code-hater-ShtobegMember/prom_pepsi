# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import traceback

# --- Quaternion Class ---
# Ensure this class definition is exactly as follows
class Quaternion:
    """Represents a quaternion for 3D rotations."""

    def init(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def repr(self):
        return f"Q(w={self.w:.3f}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    def mul(self, other):
        """Quaternion multiplication or scalar multiplication."""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float, np.number)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Multiplication only supports Quaternion or scalar values")

    def add(self, other):
        """Quaternion addition."""
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError("Addition only supports two Quaternions")

    def conjugate(self):
        """Returns the conjugate of the quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_sq(self):
        """Calculates the squared norm (magnitude squared)."""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def norm(self):
        """Calculates the norm (magnitude) of the quaternion."""
        return np.sqrt(self.norm_sq())

    def normalize(self):
        """Normalizes the quaternion to unit length."""
        n = self.norm()
        if n < 1e-9:  # Avoid division by zero
            # Return identity quaternion if norm is too small
            # print("Warning: Normalizing near-zero quaternion. Returning identity.") # Optional warning
            return Quaternion(1, 0, 0, 0)  # Calls init
        return self * (1.0 / n)

    def to_rotation_matrix(self):
        """Converts the unit quaternion to a 3x3 rotation matrix."""
        # Ensure it's normalized first
        q_norm = self.normalize()
        w, x, y, z = q_norm.w, q_norm.x, q_norm.y, q_norm.z

        # Precompute squares
        x2, y2, z2 = x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        # Rotation matrix elements
        R = np.array([
            [1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy],
            [2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx],
            [2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2]
        ])
        return R

    def rotate_vector(self, v):
        """Rotates a 3D vector using the quaternion (q * v * q_conj)."""
        v = np.asarray(v)
        if v.shape != (3,):
            raise ValueError("Input vector must be 3D")
        v_quat = Quaternion(0, v[0], v[1], v[2])  # Calls init
        # Use normalized quaternion for rotation
        q_norm = self.normalize()
        rotated_v_quat = q_norm * v_quat * q_norm.conjugate()
        return np.array([rotated_v_quat.x, rotated_v_quat.y, rotated_v_quat.z])

    @staticmethod
    def identity():
        """Returns the identity quaternion."""
        # This line calls the init method above
        return Quaternion(1, 0, 0, 0)

# --- Physics Parameters ---
# Bowling Ball (Sphere)
BALL_RADIUS = 0.109  # meters (approx 8.5 inches diameter / 2)
BALL_MASS = 5.0  # kg (approx 13.2 lbs)
BALL_INERTIA_SCALAR = (2.0 / 5.0) * BALL_MASS * BALL_RADIUS**2  # Scalar for sphere
BALL_INV_INERTIA_LOCAL = np.diag([1.0 / BALL_INERTIA_SCALAR] * 3)  # Inverse inertia tensor (local)

# Bowling Pin (Cylinder) - Approximated as solid cylinder
PIN_HEIGHT = 0.38  # meters (approx 15 inches)
PIN_RADIUS = 0.06  # meters (approx 4.75 inches diameter / 2 at widest)
PIN_MASS = 1.5  # kg (approx 3.3 lbs)
# Moment of inertia for solid cylinder (local frame: z is height axis)
Ixx_pin = (1.0 / 12.0) * PIN_MASS * (3 * PIN_RADIUS**2 + PIN_HEIGHT**2)
Iyy_pin = Ixx_pin
Izz_pin = (1.0 / 2.0) * PIN_MASS * PIN_RADIUS**2
PIN_INERTIA_LOCAL = np.diag([Ixx_pin, Iyy_pin, Izz_pin])
# Inverse Inertia Tensor (Local Frame) - handle potential zero division if needed
try:
    PIN_INV_INERTIA_LOCAL = np.linalg.inv(PIN_INERTIA_LOCAL)
except np.linalg.LinAlgError:
    print("Error: Pin inertia tensor is singular. Using pseudo-inverse.")
    PIN_INV_INERTIA_LOCAL = np.linalg.pinv(PIN_INERTIA_LOCAL)

# Simulation Parameters
DT = 0.01  # Time step (smaller for better accuracy)
NUM_FRAMES = 400  # Number of frames for animation
COLLISION_EPSILON = 0.8  # Coefficient of restitution (1.0 = perfectly elastic)
GRAVITY = np.array([0, 0, -9.81]) # Gravity vector (optional, set to 0,0,0 to disable)
# GRAVITY = np.array([0, 0, 0])  # Disable gravity for simpler collision focus

# --- State Representation ---
# State vector components:
# 0:2 - position (x, y, z)
# 3:5 - velocity (vx, vy, vz)
# 6:9 - orientation quaternion (qw, qx, qy, qz) - Ball's is tracked but unused for graphics
# 10:12 - angular velocity (wx, wy, wz) in world frame

# Initial States
# Ball starts moving towards the pin
ball_initial_pos = np.array([-1.5, 0.0, BALL_RADIUS + 0.01])  # Start on the 'ground'
ball_initial_vel = np.array([5.0, 0.1, 0.0])  # Give it some speed towards pin, slight angle
ball_initial_quat = Quaternion.identity()
ball_initial_ang_vel = np.array([0.0, 0.0, 0.0])  # No initial spin

# Pin starts stationary at origin
pin_initial_pos = np.array([0.0, 0.0, PIN_HEIGHT / 2.0 + 0.01])  # Center of mass position
pin_initial_vel = np.array([0.0, 0.0, 0.0])
pin_initial_quat = Quaternion.identity()
pin_initial_ang_vel = np.array([0.0, 0.0, 0.0])

# Combine initial states into one large vector for the integrator
initial_state = np.concatenate([
    ball_initial_pos, ball_initial_vel,
    [ball_initial_quat.w, ball_initial_quat.x, ball_initial_quat.y, ball_initial_quat.z],
    ball_initial_ang_vel,
    pin_initial_pos, pin_initial_vel,
    [pin_initial_quat.w, pin_initial_quat.x, pin_initial_quat.y, pin_initial_quat.z],
    pin_initial_ang_vel
])

STATE_LEN = len(initial_state)  # Should be 2 * (3+3+4+3) = 26
BALL_STATE_SLICE = slice(0, STATE_LEN // 2)
PIN_STATE_SLICE = slice(STATE_LEN // 2, STATE_LEN)
POS_SLICE = slice(0, 3)
VEL_SLICE = slice(3, 6)
QUAT_SLICE = slice(6, 10)
ANG_VEL_SLICE = slice(10, 13)

# --- Physics Engine ---

def get_world_inv_inertia(inv_inertia_local, orientation_q):
    """Calculates the inverse inertia tensor in world coordinates"""
    R = orientation_q.to_rotation_matrix()
    # Formula: I_inv_world = R * I_inv_local * R^T
    inv_inertia_world = R @ inv_inertia_local @ R.T
    return inv_inertia_world

def state_derivatives(t, state):
    """Calculates the time derivatives of the combined state vector"""
    derivatives = np.zeros_like(state)

    # --- Ball ---
    ball_state = state[BALL_STATE_SLICE]
    # ball_pos = ball_state[POS_SLICE] # Unused variable
    ball_vel = ball_state[VEL_SLICE]
    ball_q = Quaternion(*ball_state[QUAT_SLICE])
    ball_ang_vel = ball_state[ANG_VEL_SLICE]  # World frame angular velocity

    # Ball linear velocity derivative (acceleration)
    ball_acc = 0  # Gravity added here!
    derivatives[BALL_STATE_SLICE][VEL_SLICE] = ball_acc

    # Ball position derivative (velocity)
    derivatives[BALL_STATE_SLICE][POS_SLICE] = ball_vel

# Ball angular velocity derivative (angular acceleration = InvInertia * Torque)
    # Assuming no external torques for now
    ball_torque = np.zeros(3)
    ball_inv_inertia_world = get_world_inv_inertia(BALL_INV_INERTIA_LOCAL, ball_q)
    ball_ang_acc = ball_inv_inertia_world @ ball_torque
    derivatives[BALL_STATE_SLICE][ANG_VEL_SLICE] = ball_ang_acc

    # Ball orientation derivative (quaternion derivative)
    # dQ/dt = 0.5 * Quaternion(0, omega_world) * Q
    omega_q = Quaternion(0, *ball_ang_vel)  # Calls init
    q_dot = omega_q * ball_q * 0.5
    derivatives[BALL_STATE_SLICE][QUAT_SLICE] = [q_dot.w, q_dot.x, q_dot.y, q_dot.z]

    # --- Pin ---
    pin_state = state[PIN_STATE_SLICE]
    # pin_pos = pin_state[POS_SLICE] # Unused variable
    pin_vel = pin_state[VEL_SLICE]
    pin_q = Quaternion(*pin_state[QUAT_SLICE])
    pin_ang_vel = pin_state[ANG_VEL_SLICE]  # World frame angular velocity

    # Pin linear velocity derivative (acceleration)

    pin_acc = GRAVITY  # Gravity added here!
    derivatives[PIN_STATE_SLICE][VEL_SLICE] = pin_acc

    # Pin position derivative (velocity)
    derivatives[PIN_STATE_SLICE][POS_SLICE] = pin_vel

    # Pin angular velocity derivative (angular acceleration)
    # Assuming no external torques for now
    pin_torque = np.zeros(3)
    pin_inv_inertia_world = get_world_inv_inertia(PIN_INV_INERTIA_LOCAL, pin_q)
    pin_ang_acc = pin_inv_inertia_world @ pin_torque
    derivatives[PIN_STATE_SLICE][ANG_VEL_SLICE] = pin_ang_acc

    # Pin orientation derivative (quaternion derivative)
    omega_q_pin = Quaternion(0, *pin_ang_vel)  # Calls init
    q_dot_pin = omega_q_pin * pin_q * 0.5
    derivatives[PIN_STATE_SLICE][QUAT_SLICE] = [q_dot_pin.w, q_dot_pin.x, q_dot_pin.y, q_dot_pin.z]

    return derivatives

def rk4_step(f, y, t, dt):
    """Performs a single step of the Runge-Kutta 4th order method"""
    k1 = dt * f(t, y)
    # A simple approach is to normalize *after* the final step.
    k2 = dt * f(t + dt / 2, y + k1 / 2)
    k3 = dt * f(t + dt / 2, y + k2 / 2)
    k4 = dt * f(t + dt, y + k3)
    y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # --- Normalize Quaternions after step ---
    # Ball
    q_ball_vec = y_new[BALL_STATE_SLICE][QUAT_SLICE]
    q_ball = Quaternion(*q_ball_vec).normalize()  # Calls init
    y_new[BALL_STATE_SLICE][QUAT_SLICE] = [q_ball.w, q_ball.x, q_ball.y, q_ball.z]
    # Pin
    q_pin_vec = y_new[PIN_STATE_SLICE][QUAT_SLICE]
    q_pin = Quaternion(*q_pin_vec).normalize()  # Calls init
    y_new[PIN_STATE_SLICE][QUAT_SLICE] = [q_pin.w, q_pin.x, q_pin.y, q_pin.z]

    return y_new

def check_and_resolve_collision(state, dt):
    """Checks and resolves collisions between ball and pin"""
    ball_state = state[BALL_STATE_SLICE]
    pin_state = state[PIN_STATE_SLICE]

    ball_pos = ball_state[POS_SLICE]
    ball_vel = ball_state[VEL_SLICE]
    ball_q = Quaternion(*ball_state[QUAT_SLICE])  # Calls init
    ball_ang_vel = ball_state[ANG_VEL_SLICE]

    pin_pos = pin_state[POS_SLICE]
    pin_vel = pin_state[VEL_SLICE]
    pin_q = Quaternion(*pin_state[QUAT_SLICE])  # Calls init
    pin_ang_vel = pin_state[ANG_VEL_SLICE]

    # Vector from pin center to ball center
    vec_pin_to_ball = ball_pos - pin_pos

    # --- Broad Phase Collision Check ---
    dist_centers_sq = np.dot(vec_pin_to_ball, vec_pin_to_ball)
    # Use sum of radii + half height for a rough bounding sphere for the pin
    bounding_radius_pin = np.sqrt(PIN_RADIUS**2 + (PIN_HEIGHT / 2)**2)
    min_dist_centers = BALL_RADIUS + bounding_radius_pin
    if dist_centers_sq > min_dist_centers**2:
        return state  # Too far apart for collision~

    # --- Detailed Check (Sphere vs. Cylinder Body) ---
    pin_z_axis_world = pin_q.rotate_vector([0, 0, 1])
    pin_y_axis_world = pin_q.rotate_vector([0, 1, 0])  # Needed for wall normal if centered
    pin_x_axis_world = pin_q.rotate_vector([1, 0, 0])  # Needed for wall normal if centered

    proj_len = np.dot(vec_pin_to_ball, pin_z_axis_world)  # Projection onto pin's height axis
    # Closest point on pin's central axis to ball center
    closest_pt_on_axis = pin_pos + proj_len * pin_z_axis_world
    vec_axis_to_ball = ball_pos - closest_pt_on_axis
    dist_axis_to_ball_sq = np.dot(vec_axis_to_ball, vec_axis_to_ball)
    dist_axis_to_ball = np.sqrt(dist_axis_to_ball_sq)

    # Potential collision conditions
    is_within_height_range = abs(proj_len) <= PIN_HEIGHT / 2.0
    is_within_radial_range = dist_axis_to_ball <= PIN_RADIUS + BALL_RADIUS
    is_within_cap_height_range = abs(proj_len) > PIN_HEIGHT / 2.0 and abs(proj_len) < PIN_HEIGHT / 2.0 + BALL_RADIUS
    is_radially_inside_cap_proj = dist_axis_to_ball <= PIN_RADIUS  # Check if ball center projects inside pin radius

    collided = False
    penetration_depth = 0.0
    contact_normal = np.zeros(3)
    contact_point_ball_local = np.zeros(3)
    contact_point_pin_local = np.zeros(3)

    # Check Wall Collision
    if is_within_height_range and is_within_radial_range:
        if dist_axis_to_ball > 1e-6:
            contact_normal = vec_axis_to_ball / dist_axis_to_ball
            penetration_depth = (PIN_RADIUS + BALL_RADIUS) - dist_axis_to_ball
            if penetration_depth > 0:
                collided = True
                contact_point_ball_local = -contact_normal * BALL_RADIUS
                contact_point_pin_world = closest_pt_on_axis + contact_normal * PIN_RADIUS
                contact_point_pin_local = pin_q.conjugate().rotate_vector(contact_point_pin_world - pin_pos)
                # print("Wall Hit")
        else:  # Ball center is on the pin axis - push radially
            # Use pin's local X or Y axis transformed to world as normal
            contact_normal = pin_x_axis_world if abs(np.dot(vec_pin_to_ball, pin_x_axis_world)) > abs(np.dot(vec_pin_to_ball, pin_y_axis_world)) else pin_y_axis_world
            penetration_depth = (PIN_RADIUS + BALL_RADIUS)  # Max penetration
            collided = True
            contact_point_ball_local = -contact_normal * BALL_RADIUS
            contact_point_pin_world = closest_pt_on_axis + contact_normal * PIN_RADIUS
            contact_point_pin_local = pin_q.conjugate().rotate_vector(contact_point_pin_world - pin_pos)
            # print("Wall Hit (Centered)")

    # Check Cap Collision (only if not already hit wall)
    if not collided and is_within_cap_height_range and is_radially_inside_cap_proj:
        is_top_cap = proj_len > 0
        contact_normal = pin_z_axis_world if is_top_cap else -pin_z_axis_world
        # Penetration depth along the normal
        penetration_depth = BALL_RADIUS - (abs(proj_len) - PIN_HEIGHT / 2.0)
        if penetration_depth > 0:
            collided = True
            contact_point_ball_local = -contact_normal * BALL_RADIUS
            # Contact point on pin cap (approximated at projection point)
            contact_point_pin_world = closest_pt_on_axis  # Approx point on cap surface
            contact_point_pin_local = pin_q.conjugate().rotate_vector(contact_point_pin_world - pin_pos)
            # print("Cap Hit")

    # If no collision detected, return state unchanged
    if not collided:
        return state

    # --- Resolve Penetration ---
    total_mass = BALL_MASS + PIN_MASS
    inv_total_mass = 1.0 / total_mass if total_mass > 0 else 0
    move_fraction_ball = PIN_MASS * inv_total_mass
    move_fraction_pin = BALL_MASS * inv_total_mass

    correction = contact_normal * penetration_depth * 1.01  # Move slightly more than depth
    ball_pos += correction * move_fraction_ball
    pin_pos -= correction * move_fraction_pin
    state[BALL_STATE_SLICE][POS_SLICE] = ball_pos
    state[PIN_STATE_SLICE][POS_SLICE] = pin_pos

    # --- Calculate Impulse ---
    r_ball_world = ball_q.rotate_vector(contact_point_ball_local)
    r_pin_world = pin_q.rotate_vector(contact_point_pin_local)

    v_contact_ball = ball_vel + np.cross(ball_ang_vel, r_ball_world)
    v_contact_pin = pin_vel + np.cross(pin_ang_vel, r_pin_world)
    v_relative = v_contact_ball - v_contact_pin
    v_rel_normal = np.dot(v_relative, contact_normal)

    if v_rel_normal >= -1e-6:  # Allow small tolerance for resting contact
        # print("Objects separating or resting, no impulse needed.")
        return state

    inv_I_ball_world = get_world_inv_inertia(BALL_INV_INERTIA_LOCAL, ball_q)
    inv_I_pin_world = get_world_inv_inertia(PIN_INV_INERTIA_LOCAL, pin_q)

    # Vector part of impulse calculation involving inertia tensors
    ang_impulse_factor_ball = np.cross(inv_I_ball_world @ np.cross(r_ball_world, contact_normal), r_ball_world)
    ang_impulse_factor_pin = np.cross(inv_I_pin_world @ np.cross(r_pin_world, contact_normal), r_pin_world)
    ang_impulse_term = np.dot(ang_impulse_factor_ball + ang_impulse_factor_pin, contact_normal)

    # Denominator for impulse calculation
    impulse_denom = (1.0 / BALL_MASS) + (1.0 / PIN_MASS) + ang_impulse_term

    if abs(impulse_denom) < 1e-9:  # Avoid division by zero if denominator is tiny
        print("Warning: Impulse denominator near zero. Skipping impulse.")
        return state

    impulse_j = -(1.0 + COLLISION_EPSILON) * v_rel_normal / impulse_denom
    impulse_vector = impulse_j * contact_normal

    # --- Apply Impulse ---
    new_ball_vel = ball_vel + impulse_vector / BALL_MASS
    new_pin_vel = pin_vel - impulse_vector / PIN_MASS

    new_ball_ang_vel = ball_ang_vel + inv_I_ball_world @ np.cross(r_ball_world, impulse_vector)
    new_pin_ang_vel = pin_ang_vel - inv_I_pin_world @ np.cross(r_pin_world, impulse_vector)

    state[BALL_STATE_SLICE][VEL_SLICE] = new_ball_vel
    state[BALL_STATE_SLICE][ANG_VEL_SLICE] = new_ball_ang_vel
    state[PIN_STATE_SLICE][VEL_SLICE] = new_pin_vel
    state[PIN_STATE_SLICE][ANG_VEL_SLICE] = new_pin_ang_vel

    # print("Collision Resolved.")
    return state



def floor_collusion(state, dt):
    """Checks and resolves collisions between ball and pin"""
    pin_state = state[PIN_STATE_SLICE]


    pin_pos = pin_state[POS_SLICE]
    pin_vel = pin_state[VEL_SLICE]
    pin_q = Quaternion(*pin_state[QUAT_SLICE])  # Calls init
    pin_ang_vel = pin_state[ANG_VEL_SLICE]

    # --- Detailed Check (Sphere vs. Cylinder Body) ---
    pin_z_axis_world = pin_q.rotate_vector([0, 0, 1])
    pin_y_axis_world = pin_q.rotate_vector([0, 1, 0])  # Needed for wall normal if centered
    pin_x_axis_world = pin_q.rotate_vector([1, 0, 0])  # Needed for wall normal if centered

    proj_len = np.dot(pin_pos[2], pin_z_axis_world)  # Projection onto pin's height axis
    # Closest point on pin's central axis to ball center
    closest_pt_on_axis = pin_pos + proj_len * pin_z_axis_world
    vec_axis_to_floor = np.array([closest_pt_on_axis[0], closest_pt_on_axis[1],0]) - closest_pt_on_axis
    dist_axis_to_floor_sq = np.dot(vec_axis_to_floor, vec_axis_to_floor)
    dist_axis_to_floor = np.sqrt(dist_axis_to_floor_sq)
    contact_normal = np.array([0, 0, 1])
    penetration_depth = 0

    if closest_pt_on_axis[2] <= 1e-6:
        # --- Resolve Penetration ---

        print(2143)
        correction = contact_normal * penetration_depth * 1.01  # Move slightly more than depth
        pin_pos -= correction
        state[PIN_STATE_SLICE][POS_SLICE] = pin_pos

        # --- Calculate Impulse ---
        closest_pt_on_axis_local = pin_q.conjugate().rotate_vector(closest_pt_on_axis - pin_pos)
        r_pin_world = pin_q.rotate_vector(closest_pt_on_axis_local)

        v_contact_pin = pin_vel + np.cross(pin_ang_vel, r_pin_world)
        v_relative = - v_contact_pin
        v_rel_normal = np.dot(v_relative, contact_normal)

        # if v_rel_normal >= -1e-6:  # Allow small tolerance for resting contact
        #     # print("Objects separating or resting, no impulse needed.")
        #     return state

        inv_I_pin_world = get_world_inv_inertia(PIN_INV_INERTIA_LOCAL, pin_q)

# Vector part of impulse calculation involving inertia tensors
        ang_impulse_factor_pin = np.cross(inv_I_pin_world @ np.cross(r_pin_world, contact_normal), r_pin_world)
        ang_impulse_term = np.dot(ang_impulse_factor_pin, contact_normal)

        # Denominator for impulse calculation
        impulse_denom = (1.0 / PIN_MASS) + ang_impulse_term

        if abs(impulse_denom) < 1e-9:  # Avoid division by zero if denominator is tiny
            print("Warning: Impulse denominator near zero. Skipping impulse.")
            return state

        impulse_j = -(1.0 + COLLISION_EPSILON) * v_rel_normal / impulse_denom
        impulse_vector = impulse_j * contact_normal

        # --- Apply Impulse ---
        new_pin_vel = pin_vel - impulse_vector / PIN_MASS
        new_pin_ang_vel = pin_ang_vel - inv_I_pin_world @ np.cross(r_pin_world, impulse_vector)

        state[PIN_STATE_SLICE][VEL_SLICE] = new_pin_vel
        state[PIN_STATE_SLICE][ANG_VEL_SLICE] = new_pin_ang_vel

    # print("Collision Resolved.")
    return state
# --- Visualization ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Pre-calculate sphere mesh points (unit sphere)
u_sph, v_sph = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_sph_unit = np.cos(u_sph) * np.sin(v_sph)
y_sph_unit = np.sin(u_sph) * np.sin(v_sph)
z_sph_unit = np.cos(v_sph)

# Pre-calculate cylinder mesh points (unit radius, unit height centered at origin)
theta_cyl = np.linspace(0, 2 * np.pi, 30)
z_cyl_unit = np.linspace(-0.5, 0.5, 10)  # Unit height centered at 0
theta_cyl_grid, z_cyl_grid_unit = np.meshgrid(theta_cyl, z_cyl_unit)
x_cyl_unit = np.cos(theta_cyl_grid)  # Unit radius
y_cyl_unit = np.sin(theta_cyl_grid)
# Caps (unit radius)
theta_cap = np.linspace(0, 2 * np.pi, 30)
r_cap = np.linspace(0, 1, 5)  # Unit radius cap
theta_cap_grid, r_cap_grid = np.meshgrid(theta_cap, r_cap)
x_cap_unit = r_cap_grid * np.cos(theta_cap_grid)
y_cap_unit = r_cap_grid * np.sin(theta_cap_grid)

# Store current state globally for update function
current_state = initial_state.copy()
ani = None  # Initialize ani to None

def update(frame):
    """Updates the animation frame"""
    global current_state, ani  # Make ani accessible
    try:
        # Integrate state forward
        current_state = rk4_step(state_derivatives, current_state, frame * DT, DT)

        # Check and resolve collisions (can happen multiple times per frame if needed)
        # Basic implementation: check once per frame
        current_state = check_and_resolve_collision(current_state, DT)
        current_state = floor_collusion(current_state, DT)


        # --- Extract states for plotting ---
        ball_state = current_state[BALL_STATE_SLICE]
        pin_state = current_state[PIN_STATE_SLICE]

        ball_pos = ball_state[POS_SLICE]
        pin_pos = pin_state[POS_SLICE]
        pin_q = Quaternion(*pin_state[QUAT_SLICE])  # Calls init
        pin_rot_matrix = pin_q.to_rotation_matrix()

        # --- Plotting ---
        ax.clear()
        plot_limit = 1.5  # Adjust as needed
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_zlim(0, plot_limit)  # Start Z from 0 (ground)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Bowling Simulation (Frame {frame})')

        # Draw Ground Plane (optional)
        gx, gy = np.meshgrid(np.linspace(-plot_limit, plot_limit, 5), np.linspace(-plot_limit, plot_limit, 5))
        gz = np.zeros_like(gx)
        ax.plot_surface(gx, gy, gz, color='lightgrey', alpha=0.3, zorder=-1)

        # Draw Ball
        x_ball = BALL_RADIUS * x_sph_unit + ball_pos[0]
        y_ball = BALL_RADIUS * y_sph_unit + ball_pos[1]
        z_ball = BALL_RADIUS * z_sph_unit + ball_pos[2]
        ax.plot_surface(x_ball, y_ball, z_ball, color='darkblue', alpha=0.8)

# Draw Pin (Rotated and Translated)
        # Wall points
        cyl_points_local = np.vstack([
            (PIN_RADIUS * x_cyl_unit).flatten(),
            (PIN_RADIUS * y_cyl_unit).flatten(),
            (PIN_HEIGHT * z_cyl_grid_unit).flatten()  # Scale height
        ])
        cyl_points_world = pin_rot_matrix @ cyl_points_local + pin_pos[:, np.newaxis]
        x_cyl = cyl_points_world[0, :].reshape(x_cyl_unit.shape)
        y_cyl = cyl_points_world[1, :].reshape(y_cyl_unit.shape)
        z_cyl = cyl_points_world[2, :].reshape(z_cyl_grid_unit.shape)
        ax.plot_surface(x_cyl, y_cyl, z_cyl, color='red', alpha=0.6)

        # Top Cap
        cap_top_points_local = np.vstack([
            (PIN_RADIUS * x_cap_unit).flatten(),
            (PIN_RADIUS * y_cap_unit).flatten(),
            np.full(x_cap_unit.size, PIN_HEIGHT * 0.5)  # At scaled top height
        ])
        cap_top_points_world = pin_rot_matrix @ cap_top_points_local + pin_pos[:, np.newaxis]
        x_cap_top = cap_top_points_world[0, :].reshape(x_cap_unit.shape)
        y_cap_top = cap_top_points_world[1, :].reshape(y_cap_unit.shape)
        z_cap_top = cap_top_points_world[2, :].reshape(x_cap_unit.shape)
        ax.plot_surface(x_cap_top, y_cap_top, z_cap_top, color='white', alpha=0.7)

        # Bottom Cap
        cap_bottom_points_local = np.vstack([
            (PIN_RADIUS * x_cap_unit).flatten(),
            (PIN_RADIUS * y_cap_unit).flatten(),
            np.full(x_cap_unit.size, -PIN_HEIGHT * 0.5)  # At scaled bottom height
        ])
        cap_bottom_points_world = pin_rot_matrix @ cap_bottom_points_local + pin_pos[:, np.newaxis]
        x_cap_bottom = cap_bottom_points_world[0, :].reshape(x_cap_unit.shape)
        y_cap_bottom = cap_bottom_points_world[1, :].reshape(y_cap_unit.shape)
        z_cap_bottom = cap_bottom_points_world[2, :].reshape(x_cap_unit.shape)
        ax.plot_surface(x_cap_bottom, y_cap_bottom, z_cap_bottom, color='darkred', alpha=0.7)

        return ax,

    except Exception as e:
        print(f"Error in frame {frame}:")
        traceback.print_exc()
        # Stop animation on error
        if ani and hasattr(ani, 'event_source') and ani.event_source:
            ani.event_source.stop()
        raise

# --- Run Animation ---
# Assign to global ani variable
ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, interval=max(1, int(DT * 1000)), blit=False, repeat=False)

# --- Option: Save ---
# print("Attempting to save animation...")
# try:
#     ani.save('bowling_simulation.mp4', writer='ffmpeg', fps=int(1/DT), dpi=150)
#     print("Animation saved successfully.")
# except Exception as e:
#     print(f"Error saving animation: {e}")
#     print("Ensure ffmpeg is installed and in PATH.")

plt.show()
print("Animation finished or window closed.")