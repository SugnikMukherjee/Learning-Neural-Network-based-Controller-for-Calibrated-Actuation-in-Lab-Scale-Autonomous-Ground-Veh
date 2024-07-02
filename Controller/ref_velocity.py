import rospy
from time import time
from std_msgs.msg import Float64
import numpy as np

ref_v = 0

if __name__ == '__main__':
    rospy.init_node('ref_velocity_publisher')
    ref_vel_pub = rospy.Publisher('ref_vel', Float64, queue_size=10,latch = True)
    try:
        while not rospy.is_shutdown():
            ref_v = float(input('Enter the reference velocity:'))
            ref_vel_pub.publish(Float64(ref_v))
    except (KeyboardInterrupt, Exception, rospy.ROSException) as e:
        print(e, '\nexecution stopped ...')