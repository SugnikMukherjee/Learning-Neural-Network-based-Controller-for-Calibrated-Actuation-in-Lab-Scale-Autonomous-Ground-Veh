import sys
import math
import time
import rospy
import numpy as np
from numpy.linalg import norm
import torch 
import torch.nn as nn
from geometry_msgs.msg import Vector3, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

# Initialize global variables
v = vx = vy = yaw = ax = ay = gz = throttle_pwm = steering_pwm = 0
voltage = 8.4

# Sampling time
dt = 0.1

#Input and Output Dimensions
input_dim = 8
output_dim = 6

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


model = NeuralNetwork(96,120,64,0.1,'leaky_relu')
model.load_state_dict(torch.load('/home/jetson/jetracer/notebooks/sugnik/pm8.pth'))

def deadReckoning():
    global v,vx,vy,ax,ay,dt
    vx += ax * dt
    vy += ay * dt
    v = math.sqrt(vx**2 + vy**2)


def st_callback(msg):
    global steering_pwm
    steering_pwm = msg.data
    #rospy.sleep(dt)

def th_callback(msg):
    global throttle_pwm
    throttle_pwm = msg.data
    #rospy.sleep(dt)


if __name__ == '__main__':
    rospy.init_node('filtered_imu_data_publisher')
#     rospy.Subscriber("steering_pwm", Float64,st_callback)
#     rospy.Subscriber("throttle_pwm",Float64,th_callback)
    imu_pub = rospy.Publisher('imu/data', Imu, queue_size=10,latch = True)
    volt_pub = rospy.Publisher('voltage', Float64, queue_size=10,latch = True)
    try:
        print('started publishing imu data ...')
        msg = Imu()
        msg.header.seq = 0
        msg.header.frame_id = 'imu_link'

        while not rospy.is_shutdown():
            steering_pwm_msg = rospy.wait_for_message("steering_pwm", Float64, timeout=100)
            st_callback(steering_pwm_msg)
            throttle_pwm_msg = rospy.wait_for_message("throttle_pwm", Float64, timeout=100)
            th_callback(throttle_pwm_msg)
#             print(steering_pwm,throttle_pwm)
            output_data = model(torch.Tensor([v, yaw, ax, ay, gz, steering_pwm, throttle_pwm, voltage]))
            v, yaw, ax, ay, gz, voltage = output_data.tolist()
            accl = np.array([ax, ay,0])
            gyro = np.array([0,0,gz])

            msg.header.seq += 1
            msg.header.stamp = rospy.Time.now()
            quat = quaternion_from_euler(0,0,yaw)
            msg.orientation = Quaternion(*quat)
            msg.linear_acceleration = Vector3(*accl)
            msg.angular_velocity = Vector3(*gyro)
            imu_pub.publish(msg)
            volt_pub.publish(Float64(voltage))
            deadReckoning()
            rospy.sleep(dt)

    except (KeyboardInterrupt, Exception, rospy.ROSException) as e:
        print(e, '\nexecution stopped ...')
