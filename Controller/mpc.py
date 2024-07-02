import torch
import torch.nn as nn
from torch2trt import TRTModule
import rospy
from time import time
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
import numpy as np
import control as ct
import control.optimal as opt
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from scipy.integrate import odeint


# Initialize global variables
v = yaw = ax = ay = gz = 0
voltage = time_in = 0
v_ref = []
st_pwm = th_pwm = []

# Sampling time
# dt = 0.1

# # Define the vehicle dynamics update function
# def vehicle_update(t, x, u, params={}):
#     return np.array([u[0], u[1]])

# # Define the vehicle output function
# def vehicle_output(t, x, u, params):
#     return x

# Simulation parameters
T, dt = 10, 0.1
s0 = np.array([0, 0])
timesteps = np.arange(0, T, dt)
states = np.array([s0])
inputs = np.array([[0, 0]])  


input_dim = 5
output_dim = 2
class NeuralNetwork(nn.Module):
  def __init__(self,layer_1,layer_2,layer_3,dropout_rate,activation):
    super(NeuralNetwork,self).__init__()
    self.activation = activation
    self.relu_nn_model = nn.Sequential(
        nn.Linear(input_dim,layer_1),
        nn.ReLU(),
        nn.Linear(layer_1,layer_2),
        nn.ReLU(),
        nn.Dropout(p = dropout_rate),
        nn.Linear(layer_2,layer_3),
        nn.ReLU(),
        nn.Linear(layer_3,output_dim)
    )
    self.leaky_relu_nn_model = nn.Sequential(
        nn.Linear(input_dim,layer_1),
        nn.LeakyReLU(),
        nn.Linear(layer_1,layer_2),
        nn.LeakyReLU(),
        nn.Dropout(p = dropout_rate),
        nn.Linear(layer_2,layer_3),
        nn.LeakyReLU(),
        nn.Linear(layer_3,output_dim)
    )
    self.tanh_nn_model = nn.Sequential(
        nn.Linear(input_dim,layer_1),
        nn.Tanh(),
        nn.Linear(layer_1,layer_2),
        nn.Tanh(),
        nn.Dropout(p = dropout_rate),
        nn.Linear(layer_2,layer_3),
        nn.Tanh(),
        nn.Linear(layer_3,output_dim)
    )
  def forward(self,x):
    if self.activation == 'relu':
      y = self.relu_nn_model(x)
      return y
    if self.activation == 'leaky_relu':
      y = self.leaky_relu_nn_model(x)
      return y
    if self.activation == 'tanh':
      y = self.tanh_nn_model(x)
      return y

def dynamics(state, t, u):
    A = np.diag([0, 0])
    B = np.eye(2)
    return A @ state + B @ u

# Callback to update IMU data
def update_from_imu(msg):
    global ax, ay, gz, yaw, v
    ax = msg.linear_acceleration.x
    ay = msg.linear_acceleration.y
    gz = msg.angular_velocity.z
    o = msg.orientation
    q = [o.x, o.y, o.z, o.w]
    yaw = euler_from_quaternion(q)[-1]
    v = np.sqrt(ax ** 2 + ay ** 2) * dt
#     rospy.sleep(dt)

# Callback to update voltage data
def update_voltage(msg):
    global voltage
    voltage = msg.data
#     rospy.sleep(dt)

def update_ref_v(msg):
    global v_ref
    v_ref.append(msg.data)
#     rospy.sleep(dt)
    
if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('car_driver_pid')

    # Subscribe to the IMU and voltage topics
#     rospy.Subscriber("imu/data", Imu, update_from_imu)
#     rospy.Subscriber("voltage", Float64, update_voltage)

    # Publisher for the neural network input data
    st_pub = rospy.Publisher("steering_pwm", Float64, queue_size=10, latch=True)
    th_pub = rospy.Publisher("throttle_pwm",Float64,queue_size=10, latch=True)
    st_pub.publish(Float64(0))
    th_pub.publish(Float64(0))
    # Load the TensorRT model
#     model_trt = TRTModule()
    model_trt = NeuralNetwork(40,72,80,0.2,'relu')
    model_trt.load_state_dict(torch.load('/home/jetson/jetracer/notebooks/sugnik/pm7.pth'))
#     model_trt.load_state_dict(torch.load('jetracer/notebooks/sugnik/pm7.pth'))
    print('\tloaded nn model ....')
    # Define the vehicle system
#     vehicle = ct.NonlinearIOSystem(
#         vehicle_update, vehicle_output, states=2, name='vehicle',
#         inputs=('acceleration', 'yaw_rate'), outputs=('velocity', 'yaw')
#     )

#     # Initial and final states and inputs
#     x0 = np.array([0., 0.])
#     u0 = np.array([0, 0])
# #     xf = np.array([5, 0])
# #     uf = np.array([0, 0])

#     # Define cost matrices
#     Q = np.diag([0.1, 0.01])  # State cost
#     R = np.diag([10, 100])    # Input cost
#     P = np.diag([100, 100])   # Final state cost

#     # Simulation parameters
#     horizon = 10
#     Tf, dt = 40, 0.1
#     timepts = np.arange(0, Tf, dt)
#     xs, us = x0.copy().reshape(1, -1), [[0, 0]]

#     # Main control loop
#     for i in range(1, len(timepts)):
#         imu_msg = rospy.wait_for_message("imu/data", Imu, timeout=10)
#         update_from_imu(imu_msg)
#         voltage_msg = rospy.wait_for_message("voltage", Float64, timeout=10)
#         update_voltage(voltage_msg)
#         xf = np.array([v, yaw])
#         # Define trajectory and terminal cost
#         traj_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=us[-1])
#         term_cost = opt.quadratic_cost(vehicle, P, 0, x0=xf)
#         start = time()
#         result = opt.solve_ocp(vehicle, np.arange(horizon) * dt, xs[-1], traj_cost, terminal_cost=term_cost, initial_guess=us[-1])
#         end = time()
#         print(f'\ttimestep - {i}\t|\texecution time = {end - start:.3f}')
#         t = [timepts[i - 1], timepts[i]]
#         us = np.vstack([us, result.inputs.T[0]])
#         x_ = odeint(vehicle_update, xs[-1], t, args=(us[-1],), tfirst=True)[-1]
#         xs = np.vstack([xs, x_])
    # System and LQR parameters
    A = np.diag([0, 0])
    B = np.eye(2)
    Q = np.diag([10000, 1])
    R = np.diag([10000, 1])
    K, S, _ = ct.lqr(A, B, Q, R)                                                                                                                                       
    try:
        while not rospy.is_shutdown(): 
            ref_v_msg = rospy.wait_for_message("ref_vel",Float64,timeout = 10)
            update_ref_v(ref_v_msg)
            imu_msg = rospy.wait_for_message("imu/data", Imu, timeout=100)
            update_from_imu(imu_msg)
            voltage_msg = rospy.wait_for_message("voltage", Float64, timeout=100)
            update_voltage(voltage_msg)
            u = -K @ np.subtract(s0, [v_ref[-1], math.sin(time_in)])
            s0 = odeint(dynamics, s0, [0, dt], args=(u,))[-1]
            states = np.vstack([states, s0])
            inputs = np.vstack([inputs, u])
            # Use the neural network model for predictions
            prev_acc = 0 if ax == 0 or ay == 0 else math.sqrt(ax ** 2 + ay ** 2) * (1 if (ax * ay) > 0 else -1)
            y_pred = model_trt(torch.Tensor([prev_acc, inputs[-1][0],inputs[-1][1],gz,voltage]))
            model_output = y_pred.cpu().detach().numpy()
            print(model_output)
            st_pub.publish(Float64(model_output[1]))
            th_pub.publish(Float64(model_output[0]))
            st_pwm.append(model_output[1])
            th_pwm.append(model_output[0])
            time_in += dt
            rospy.sleep(dt)
        plt.figure(figsize=(10,10))
        plt.subplot(611)
        plt.plot(states.T[0], '--.',label = 'Velocity')
        plt.plot(v_ref,label = 'v_ref')
        plt.legend()
        plt.subplot(612)
        plt.plot(states.T[1], '--.',label = 'Yaw')
        plt.legend()
        plt.subplot(613)
        plt.plot(inputs[:, 0],label = 'Acceleration')
        plt.legend()
        plt.subplot(614)
        plt.plot(inputs[:, 1],label = 'Yaw Rate')
        plt.legend()
        plt.show()
        plt.subplot(615)
        plt.plot(st_pwm,label = 'Steering pwm')
        plt.legend()
        plt.show()
        plt.subplot(616)
        plt.plot(th_pwm,label = 'Throttle pwm')
        plt.legend()
        plt.show()
        plt.savefig('file_name.png')
        print('Completed Plotting....')
#         fig,axis = plt.subplots(nrows = 4,ncols = 1,figsize=(10,10))
#         line = []
#         for i,ax in enumerate(axis):
#             if i == 0:
#                 line_t, = ax.plot([],[],lw = 3,label = 'velocity')
#                 ax_n = ax.twinx()
#                 line_t_n = ax_n.plot([],[],lw =3,label = 'ref_vel',color = 'orange')[0]
#                 line.append(line_t)
#                 line.append(line_t_n)
#                 ax.set_xlim(0,10)
#                 ax.set_ylim(-10,10)
#                 ax.grid()
#                 ax_n.set_xlim(0,10)
#                 ax_n.set_ylim(-10,10)
#                 ax_n.grid()
#                 ax.legend([line_t, line_t_n], [line_t.get_label(), line_t_n.get_label()], loc=0)
#             else:
#                 line_t, = ax.plot([],[],lw = 3)
#                 line.append(line_t)
#                 ax.set_xlim(0,10)
#                 ax.set_ylim(-10,10)
#                 ax.grid()
#         def animate(frame):
#             x = np.arange(len(states.T[0]))[:frame]
#             y1 = states.T[0][:frame]
#             y2 = v_ref[:frame]
#             y3 = states.T[1][:frame]
#             y4 = inputs[:,0][:frame]
#             y5 = inputs[:,1][:frame]
#             line[0].set_data(x,y1)
#             line[1].set_data(x,y2)
#             line[2].set_data(x,y3)
#             line[3].set_data(x,y4)
#             line[4].set_data(x,y5)
#             return line
#         anim = FuncAnimation(fig, animate,frames = len(timesteps), interval = 1,blit = True)
#         anim.save('simulate5.gif',fps = 30)
        

    except (KeyboardInterrupt, Exception, rospy.ROSException) as e: 
        plt.figure(figsize=(10,10))
        plt.subplot(611)
        plt.plot(states.T[0], '--.',label = 'Velocity')
        plt.plot(v_ref,label = 'v_ref')
        plt.legend()
        plt.subplot(612)
        plt.plot(states.T[1], '--.',label = 'Yaw')
        plt.legend()
        plt.subplot(613)
        plt.plot(inputs[:, 0],label = 'Acceleration')
        plt.legend()
        plt.subplot(614)
        plt.plot(inputs[:, 1],label = 'Yaw Rate')
        plt.legend()
        plt.subplot(615)
        plt.plot(st_pwm,label = 'Steering pwm')
        plt.legend()
        plt.show()
        plt.subplot(616)
        plt.plot(th_pwm,label = 'Throttle pwm')
        plt.legend()
        plt.show()
        plt.savefig('file_name.png')
        print('Completed Plotting....')
#         fig,axis = plt.subplots(nrows = 4,ncols = 1,figsize=(10,10))
#         line = []
#         for i,ax in enumerate(axis):
#             if i == 0:
#                 line_t, = ax.plot([],[],lw = 3,label = 'velocity')
#                 ax_n = ax.twinx()
#                 line_t_n = ax_n.plot([],[],lw =3,label = 'ref_vel',color = 'orange')[0]
#                 line.append(line_t)
#                 line.append(line_t_n)
#                 ax.set_xlim(0,10)
#                 ax.set_ylim(-10,10)
#                 ax.grid()
#                 ax_n.set_xlim(0,10)
#                 ax_n.set_ylim(-10,10)
#                 ax_n.grid()
#                 ax.legend([line_t, line_t_n], [line_t.get_label(), line_t_n.get_label()], loc=0)
#             else:
#                 line_t, = ax.plot([],[],lw = 3)
#                 line.append(line_t)
#                 ax.set_xlim(0,10)
#                 ax.set_ylim(-10,10)
#                 ax.grid()
#         def animate(frame):
#             x = np.arange(len(states.T[0]))[:frame]
#             y1 = states.T[0][:frame]
#             y2 = v_ref[:frame]
#             y3 = states.T[1][:frame]
#             y4 = inputs[:,0][:frame]
#             y5 = inputs[:,1][:frame]
#             line[0].set_data(x,y1)
#             line[1].set_data(x,y2)
#             line[2].set_data(x,y3)
#             line[3].set_data(x,y4)
#             line[4].set_data(x,y5)
#             return line
#         anim = FuncAnimation(fig, animate,frames = len(timesteps), interval = 1,blit = True)
#         anim.save('simulate5.gif',fps = 30)
        print(e, '\nexecution stopped ...')
#     rospy.spin()