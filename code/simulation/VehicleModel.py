import jax
import jax.numpy as jnp
import numpy as np

from simulation.utils import R_z, rotate_2D_Vector

rng = np.random.default_rng(987654321)

## default model parameters
default_params = {
    "c_1x": 2.5e7,
    "c_2x": 3.6e6,
    "c_1y": 1.9e6,
    "c_2y": 1.35e5,
    "C_x": 1.42,
    "C_y": 1.9,
    "E_x": -9.75,
    "E_y": 0.52,
    "C_roll1": 0.0083,
    "C_roll2": 0.0005,
    "mass": 1720,
    "Inertia_z": 2066,
    "Inertia_tire": 28.6,
    "Inertia_engine": 0.197,
    "air_resistance": 0.27,
    "dt": 0.01,
}

## constant parameters
width_front = 1.543
width_rear = 1.513
length_front = 1.16
length_rear = 1.47
radius_tire = 0.3116  # Increasing it by a 3rd should work but check the paper first and tune the value
height_CoG = 0.2684
vel_limit = 1
gravitation = 9.81

air_density = 1.225
body_surface_x = 2.19


################################################################################
### Model functions


def MTF_x(slip, F_z, max_friction, **params):
    """
    Evaluates the Magic Tire Formular.
    """

    C_F = params["c_1x"] * jnp.sin(2 * jnp.arctan(F_z / params["c_2x"]))
    D = max_friction * F_z
    B = C_F / params["C_x"] / D

    Bs = B * slip
    E = params["E_x"]
    C = params["C_x"]

    return D * jnp.sin(C * jnp.arctan((1 - E) * Bs + E * jnp.arctan(Bs)))


def MTF_y(slip, F_z, max_friction, **params):
    """
    Evaluates the Magic Tire Formular.
    """
    C_F = params["c_1y"] * jnp.sin(2 * jnp.arctan(F_z / params["c_2y"]))
    D = max_friction * F_z
    B = C_F / params["C_y"] / D

    Bs = B * slip
    E = params["E_y"]
    C = params["C_y"]

    return D * jnp.sin(C * jnp.arctan((1 - E) * Bs + E * jnp.arctan(Bs)))


def breaking_length(vel, friction, abs_efficiency):
    """
    Calculate breaking length as result of dissipated kinetic energy.
    """
    vel = np.linalg.norm(vel, axis=-1)

    return vel**2 / (2 * friction * gravitation * abs_efficiency)


def V_Regularization(vel, vel_limit):
    """
    Regularization function for vehicle speed to enforce smooth lower bound.
    """

    e = jnp.exp(-0.5 * vel**2 / vel_limit**2)

    return jnp.sqrt(vel**2 + vel_limit**2) * e + jnp.abs(vel) * (1 - e)


def tire_slip(v_x_tire, v_y_tire, tire_rate):
    """
    Calculate slip with longitudinal and lateral tire velocity
    """

    # v_reg = V_Regularization(v_x_tire, vel_limit)
    v_reg = jnp.maximum(jnp.abs(v_x_tire), vel_limit)

    # longitudinal slip
    s_x = (tire_rate * radius_tire - v_x_tire) / v_reg

    # lateral slip tan(side_slip_angle)
    s_y = -v_y_tire / v_reg

    return s_x, s_y


def static_tire_load(**params):
    """
    Calculate static tire load.
    """

    length = length_front + length_rear
    mass = params["mass"]

    m_front = length_rear * mass / length
    m_rear = length_front * mass / length

    F_z1 = gravitation / 2 * m_front
    F_z2 = gravitation / 2 * m_front

    F_z3 = gravitation / 2 * m_rear
    F_z4 = gravitation / 2 * m_rear

    return jnp.array([F_z1, F_z2, F_z3, F_z4])


def dynamic_tire_load(roll, pitch, droll, dpitch, **params):
    """
    Calculate dynamic tire load including suspension effects.
    """
    # Static load distribution
    F_z_static = static_tire_load(**params)

    c = params["suspension_stiffness"]
    d = params["suspension_damping"]

    # Calculate suspension displacement and velocity at each corner
    # Front left
    z_fl = roll * width_front / 2 + pitch * length_front
    dz_fl = droll * width_front / 2 + dpitch * length_front

    # Front right
    z_fr = -roll * width_front / 2 + pitch * length_front
    dz_fr = -droll * width_front / 2 + dpitch * length_front

    # Rear left
    z_rl = roll * width_rear / 2 - pitch * length_rear
    dz_rl = droll * width_rear / 2 - dpitch * length_rear

    # Rear right
    z_rr = -roll * width_rear / 2 - pitch * length_rear
    dz_rr = -droll * width_rear / 2 - dpitch * length_rear

    # Calculate suspension forces at each corner (spring + damper)
    F_z_dynamic = jnp.array(
        [
            F_z_static[0] + (c * z_fl + d * dz_fl),  # Front left
            F_z_static[1] + (c * z_fr + d * dz_fr),  # Front right
            F_z_static[2] + (c * z_rl + d * dz_rl),  # Rear left
            F_z_static[3] + (c * z_rr + d * dz_rr),  # Rear right
        ]
    )

    return jnp.maximum(F_z_dynamic, 0.0)  # Ensure non-negative loads


def tire_velocity(v_x_CoG, v_y_CoG, yaw_rate, steer_ang):
    """
    Calculate longitudinal and lateral tire velocity in tire frame.
    """

    v_x_fl = v_x_CoG - yaw_rate * width_front / 2
    v_y_fl = v_y_CoG + yaw_rate * length_front

    v_x_fr = v_x_CoG + yaw_rate * width_front / 2
    v_y_fr = v_y_CoG + yaw_rate * length_front

    v_x_rl = v_x_CoG - yaw_rate * width_rear / 2
    v_y_rl = v_y_CoG - yaw_rate * length_rear

    v_x_rr = v_x_CoG + yaw_rate * width_rear / 2
    v_y_rr = v_y_CoG - yaw_rate * length_rear

    vel_tire = jnp.array(
        [
            rotate_2D_Vector(v_x_fl, v_y_fl, -steer_ang[0]),
            rotate_2D_Vector(v_x_fr, v_y_fr, -steer_ang[1]),
            rotate_2D_Vector(v_x_rl, v_y_rl, -steer_ang[2]),
            rotate_2D_Vector(v_x_rr, v_y_rr, -steer_ang[3]),
        ]
    )

    return vel_tire


def dynamic_tire_load_3D(a_x, a_y, a_z, **params):
    """
    Calculate dynamic tire loads based on 3D accelerations.

    Args:
        a_x: Longitudinal acceleration [m/s^2]
        a_y: Lateral acceleration [m/s^2]
        a_z: Vertical acceleration [m/s^2]
        **params: Vehicle parameters including:
            - mass: Vehicle mass [kg]

    Returns:
        Array of dynamic tire loads [N] for [FL, FR, RL, RR]
    """
    # Unpack parameters
    mass = params["mass"]

    # Calculate effective mass distribution based on accelerations
    # Similar to static_tire_load but using accelerations instead of gravity
    length = length_front + length_rear

    # Calculate mass distribution considering longitudinal acceleration
    m_front = mass / a_z * (a_z * length_rear / length + a_x * height_CoG / length)
    m_rear = mass / a_z * (a_z * length_front / length - a_x * height_CoG / length)

    # Calculate lateral mass transfer
    lat_transfer_front = mass * a_y * height_CoG / width_front
    lat_transfer_rear = mass * a_y * height_CoG / width_rear

    # Calculate tire loads with all effects
    F_z1 = m_front * a_z / 2 - lat_transfer_front  # Front left
    F_z2 = m_front * a_z / 2 + lat_transfer_front  # Front right
    F_z3 = m_rear * a_z / 2 - lat_transfer_rear  # Rear left
    F_z4 = m_rear * a_z / 2 + lat_transfer_rear  # Rear right

    return jnp.maximum(
        jnp.array([F_z1, F_z2, F_z3, F_z4]), 0.0
    )  # Ensure non-negative loads


def vehicle_dx(
    veh_state,  # [geo_pos(2), yaw, dyaw, veh_vel(2), tire_rate(4)],
    aux_input,
    p_inf,
    **params,
):
    # Unpack p_inf
    friction, air_resistance, mass = p_inf

    params["friction"] = friction
    params["air_resistance"] = air_resistance
    params["mass"] = mass

    # unpack parameters
    air_resistance = params["air_resistance"]
    mass = params["mass"]
    Inertia_z = params["Inertia_z"]
    Inertia_tire = params["Inertia_tire"]
    Inertia_engine = params["Inertia_engine"]
    C_roll1 = params["C_roll1"]
    C_roll2 = params["C_roll2"]

    # unpack state
    yaw = veh_state[2]
    dyaw = veh_state[3]
    veh_vel = veh_state[4:6]
    tire_rate = veh_state[6:10]

    # unpack aux input
    steer_ang = aux_input["steer_ang"]
    steer_ang = jnp.array([steer_ang, steer_ang, 0, 0])
    engine_torque = aux_input["engine_torque"]
    engine_torque = jnp.array([engine_torque / 2, engine_torque / 2, 0, 0])
    break_torque = aux_input["break_torque"]
    break_torque = jnp.array([break_torque, break_torque, break_torque, break_torque])
    gear_transmission = aux_input["gear_transmission"]
    gear_transmission = jnp.array([gear_transmission, gear_transmission, 0, 0])

    # contact forces with dynamic tire loads
    vel_tire = tire_velocity(veh_vel[0], veh_vel[1], dyaw, steer_ang)
    s_x, s_y = tire_slip(vel_tire[:, 0], vel_tire[:, 1], tire_rate)
    F_z = static_tire_load(**params)
    s = jnp.sqrt(s_x**2 + s_y**2) + 1e-6
    F_x_tire = MTF_x(s, F_z, friction, **params) * s_x / s
    F_y_tire = MTF_y(s, F_z, friction, **params) * s_y / s

    # rotate tire forces from tires to two-track model
    F_xy = jax.vmap(rotate_2D_Vector)(F_x_tire, F_y_tire, steer_ang)
    F_x = F_xy[:, 0]
    F_y = F_xy[:, 1]
    del F_xy

    # air resisance force
    F_air = (
        0.5
        * air_resistance
        * air_density
        * body_surface_x
        * veh_vel[0] ** 2
        * jnp.sign(veh_vel[0])
    )

    # translational dynamics
    F = jnp.array([jnp.sum(F_x) - F_air, jnp.sum(F_y)])
    dveh_vel = jnp.array(
        [
            F[0] / mass + dyaw * veh_vel[1],
            F[1] / mass - dyaw * veh_vel[0],
        ]
    )

    # yaw dynamics
    ddyaw = (
        (F_y[0] + F_y[1]) * length_front
        - (F_y[2] + F_y[3]) * length_rear
        + (F_x[0] - F_x[1]) * width_front / 2
        + (F_x[2] - F_x[3]) * width_rear / 2
    ) / Inertia_z

    # geo position movement
    dgeo_pos = (R_z(yaw).T @ jnp.array([veh_vel[0], veh_vel[1], 0]))[:2]

    ## tire angular acceleration
    roll_resistance = (
        F_z * (C_roll1 + C_roll2 * jnp.abs(tire_rate)) * jnp.sign(tire_rate)
    )

    # Calculate net torque excluding friction forces
    drive_torque = engine_torque * gear_transmission
    resistance_torque = (
        F_x_tire * radius_tire + roll_resistance + break_torque * jnp.sign(tire_rate)
    )

    # Vectorized condition over all 4 tires
    inertia = Inertia_tire + Inertia_engine / 2 * gear_transmission**2
    dtire_rate = (drive_torque - resistance_torque) / inertia

    # Concatenate all derivatives into single array
    dstate = jnp.concatenate(
        [
            dgeo_pos,  # 2
            jnp.array([dyaw]),  # 1
            jnp.array([ddyaw]),  # 1
            dveh_vel,  # 2
            dtire_rate,  # 4
        ]
    )

    return dstate


def vehicle_RK4x(
    veh_state,  # [geo_pos(2), yaw, dyaw, veh_vel(2), tire_rate(4)],
    aux_input,
    p_inf,
    **params,
):
    """
    Recursive application of Runge-Kutta-4 for virtual shortening of the time step.
    """

    n = 15
    dt = params["dt"]

    def vehicle_RK4(i, veh_state):
        # compute RK4 slopes using updated intermediary states:
        k1 = vehicle_dx(veh_state, aux_input, p_inf, **params)
        state_k2 = veh_state + dt / n / 2 * k1
        k2 = vehicle_dx(state_k2, aux_input, p_inf, **params)
        state_k3 = veh_state + dt / n / 2 * k2
        k3 = vehicle_dx(state_k3, aux_input, p_inf, **params)
        state_k4 = veh_state + dt / n * k3
        k4 = vehicle_dx(state_k4, aux_input, p_inf, **params)

        # update the state using the weighted sum of slopes
        return veh_state + dt / n / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.lax.fori_loop(0, n, vehicle_RK4, veh_state)


def vehicle_Ex(
    veh_state,  # [geo_pos(2), yaw, dyaw, veh_vel(2), tire_rate(4)],
    aux_input,
    p_inf,
    **params,
):
    """
    Recursive application of Euler method for virtual shortening of the time step.
    """

    n = 20
    dt = params["dt"]

    def vehicle_E(i, veh_state):
        dstate = vehicle_dx(veh_state, aux_input, p_inf, **params)
        return veh_state + dt / n * dstate

    return jax.lax.fori_loop(0, n, vehicle_E, veh_state)


def vehicle_fy(
    veh_state,  # [geo_pos(2), yaw, dyaw, veh_vel(2), tire_rate(4)]
    aux_input,
    p_inf,
    **params,
):

    # Unpack p_inf
    friction, air_resistance, mass = p_inf

    params["friction"] = friction
    params["air_resistance"] = air_resistance
    params["mass"] = mass

    # unpack state
    geo_pos = veh_state[0:2]
    yaw = veh_state[2]
    dyaw = veh_state[3]
    veh_vel = veh_state[4:6]
    tire_rate = veh_state[6:10]

    # unpack aux input
    steer_ang = aux_input["steer_ang"]
    steer_ang = jnp.array([steer_ang, steer_ang, 0, 0])
    engine_torque = aux_input["engine_torque"]
    engine_torque = jnp.array([engine_torque / 2, engine_torque / 2, 0, 0])
    break_torque = aux_input["break_torque"]
    break_torque = jnp.array([break_torque, break_torque, break_torque, break_torque])
    gear_transmission = aux_input["gear_transmission"]
    gear_transmission = jnp.array([gear_transmission, gear_transmission, 0, 0])

    # unpack parameters
    air_resistance = params["air_resistance"]
    mass = params["mass"]

    # contact forces
    vel_tire = tire_velocity(veh_vel[0], veh_vel[1], dyaw, steer_ang)
    s_x, s_y = tire_slip(vel_tire[:, 0], vel_tire[:, 1], tire_rate)
    F_z = static_tire_load(**params)
    s = jnp.sqrt(s_x**2 + s_y**2) + 1e-6
    F_x_tire = MTF_x(s, F_z, friction, **params) * s_x / s
    F_y_tire = MTF_y(s, F_z, friction, **params) * s_y / s

    # rotate tire forces from tires to two-track model
    F_xy = jax.vmap(rotate_2D_Vector)(F_x_tire, F_y_tire, steer_ang)
    F_x = F_xy[:, 0]
    F_y = F_xy[:, 1]
    del F_xy

    # air resisance force
    F_air = (
        0.5
        * air_resistance
        * air_density
        * body_surface_x
        * veh_vel[0] ** 2
        * jnp.sign(veh_vel[0])
    )

    # translational dynamics
    F = jnp.array([jnp.sum(F_x) - F_air, jnp.sum(F_y)])
    dveh_vel = jnp.array(
        [
            F[0] / mass + dyaw * veh_vel[1],
            F[1] / mass - dyaw * veh_vel[0],
        ]
    )

    # Return measurements as single array
    return jnp.concatenate(
        [
            # geo_pos,  # 2
            jnp.array([dyaw]),  # 1
            (R_z(yaw).T @ jnp.array([veh_vel[0], veh_vel[1], 0]))[:2],  # 2
            dveh_vel,  # 2
            tire_rate,  # 4
        ]
    )


def vehicle_measurement_equation(
    veh_state,  # [geo_pos(2), yaw, dyaw, veh_vel(2), tire_rate(4)]
    aux_input,
    p_inf,  # [friction, air_resistance, mass]
    **params,
):

    # Unpack p_inf
    friction, air_resistance, mass = p_inf

    params["friction"] = friction
    params["air_resistance"] = air_resistance
    params["mass"] = mass

    dx = vehicle_dx(veh_state, aux_input, p_inf, **default_params)
    measurement = np.array(
        [dx[2], dx[4], dx[5], veh_state[6], veh_state[7], veh_state[8], veh_state[9]]
    )

    return measurement
