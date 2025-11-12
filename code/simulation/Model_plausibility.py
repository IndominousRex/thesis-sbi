import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from VehicleModel import default_params, vehicle_Ex, vehicle_RK4x, vehicle_dx, vehicle_measurement_equation

plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


dt = default_params["dt"]
time = np.arange(0, 100, dt)

# control input
steer_angle = np.zeros((time.shape[0]))
engine_torque = np.zeros((time.shape[0]))
break_torque = np.zeros((time.shape[0]))
gear_transmission = np.ones((time.shape[0]))

torque = 1000 * np.sin(np.pi / 10 * np.arange(0, 10, dt))
engine_torque[0 : torque.size] = torque
engine_torque[torque.size + int(10 / dt) : torque.size + int(10 / dt) + torque.size] = (
    -torque * 0.5
)

steer_angle[int(10 / dt) : int(20 / dt)] = 5 / 180 * np.pi
steer_angle[int(30 / dt) : int(40 / dt)] = -5 / 180 * np.pi


# initialize states
# [geo_pos(3), yaw, dyaw, veh_vel(2), tire_rate(4)]
state_E = np.zeros(10)
state_RK4 = np.zeros(10)
friction = 0.9


aux_input = {
    "steer_ang": steer_angle[0],
    "engine_torque": engine_torque[0],
    "break_torque": break_torque[0],
    "gear_transmission": gear_transmission[0],
    "roll": 0,
    "pitch": 0,
}


# vars for logging
geo_pos_E = np.zeros((time.size, 2))
yaw_E = np.zeros((time.size))
dyaw_E = np.zeros((time.size))
veh_vel_E = np.zeros((time.size, 2))
tire_rate_E = np.zeros((time.size, 4))

geo_pos_RK4 = np.zeros((time.size, 2))
yaw_RK4 = np.zeros((time.size))
dyaw_RK4 = np.zeros((time.size))
veh_vel_RK4 = np.zeros((time.size, 2))
tire_rate_RK4 = np.zeros((time.size, 4))

ax = np.zeros((time.size))
ay = np.zeros((time.size))
yaw = np.zeros((time.size))

p_inf = (friction, 0.27, 1720)

# simulation
Ex_model = jax.jit(vehicle_Ex)
RK4_model = jax.jit(vehicle_RK4x)
measurement_model = jax.jit(vehicle_measurement_equation)
for i in tqdm(range(1, time.size)):
    # evaluate vehicle model with Euler
    state_E = Ex_model(state_E, aux_input, p_inf, **default_params)

    # evaluate vehicle model with Runge-Kutta-4
    state_RK4 = RK4_model(state_RK4, aux_input, p_inf, **default_params)

    measurement = measurement_model(state_RK4,aux_input,p_inf,**default_params)


    # update control input
    aux_input["steer_ang"] = steer_angle[i]
    aux_input["engine_torque"] = engine_torque[i]
    aux_input["break_torque"] = break_torque[i]
    aux_input["gear_transmission"] = gear_transmission[i]

    # log states
    geo_pos_E[i, :] = state_E[0:2]
    yaw_E[i] = state_E[2]
    dyaw_E[i] = state_E[3]
    veh_vel_E[i, :] = state_E[4:6]
    tire_rate_E[i, :] = state_E[6:10]

    geo_pos_RK4[i, :] = state_RK4[0:2]
    yaw_RK4[i] = state_RK4[2]
    dyaw_RK4[i] = state_RK4[3]
    veh_vel_RK4[i, :] = state_RK4[4:6]
    tire_rate_RK4[i, :] = state_RK4[6:10]

    yaw[i] = measurement[0]
    ax[i] = measurement[1]
    ay[i] = measurement[2]
    



fig_vel, ax_vel = plt.subplots(2, 1)
ax_vel[0].plot(time, veh_vel_E[:, 0])
ax_vel[0].plot(time, veh_vel_RK4[:, 0], linestyle="--")
ax_vel[1].plot(time, veh_vel_E[:, 1])
ax_vel[1].plot(time, veh_vel_RK4[:, 1], linestyle="--")
ax_vel[1].legend(["Euler", "RK4"])
ax_vel[0].set_title("$v_x$")
ax_vel[1].set_title("$v_y$")


fig_engine, ax_engine = plt.subplots(2, 1)
ax_engine[0].plot(time, engine_torque)
ax_engine[1].plot(time, steer_angle * 180 / np.pi)


fig_euler, ax_euler = plt.subplots(2, 1)
ax_euler[0].plot(time, yaw_E * 180 / np.pi)
ax_euler[0].set_title("Yaw")
ax_euler[1].plot(time, dyaw_E * 180 / np.pi)
ax_euler[1].set_title("dyaw")


fig_tireRate, ax_tireRate = plt.subplots(4, 1)
ax_tireRate[0].plot(time, tire_rate_E[:, 0])
ax_tireRate[0].plot(time, tire_rate_RK4[:, 0], linestyle="--")
ax_tireRate[1].plot(time, tire_rate_E[:, 1])
ax_tireRate[1].plot(time, tire_rate_RK4[:, 1], linestyle="--")
ax_tireRate[2].plot(time, tire_rate_E[:, 2])
ax_tireRate[2].plot(time, tire_rate_RK4[:, 2], linestyle="--")
ax_tireRate[3].plot(time, tire_rate_E[:, 3])
ax_tireRate[3].plot(time, tire_rate_RK4[:, 3], linestyle="--")
ax_tireRate[3].legend(["Euler", "RK4"])
ax_tireRate[0].set_title("Tire Rates")


fig_pos, ax_pos = plt.subplots()
ax_pos.plot(geo_pos_E[:, 0], geo_pos_E[:, 1])
ax_pos.plot(geo_pos_RK4[:, 0], geo_pos_RK4[:, 1], linestyle="--")


fig_meas, ax_meas = plt.subplots(3, 1)
ax_meas[0].plot(time, ax)
ax_meas[1].plot(time, ay)
ax_meas[2].plot(time, yaw)

plt.show()
