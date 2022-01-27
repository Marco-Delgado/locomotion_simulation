import rospy
from sensor_msgs.msg import Image as msg_Image
from nav_msgs.msg import Odometry as pose_msg
import sys
import os
import datetime
import numpy as np
import imageio
import cv2
from PIL import Image as pillow
from PIL import ImageFilter

### imports from other 2 in order use both ros and python apis

import sys

ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
sys.path.append(ros_path)
from cv_bridge import CvBridge, CvBridgeError

MIN_DEPTH = 0.0 * 1000
MAX_DEPTH = 10.0 * 1000


class ImagePoseListener:
    def __init__(self, depth_topic, color_topic):
        self.depth_topic = depth_topic
        self.color_topic = color_topic
        self.poses = []
        self.rpy = []
        self.frames = []
        self.color_frames = []
        self.curr_depth = []
        self.curr_color = []
        self.curr_pose = []
        self.count = 0
        self.color_count = 0
        self.cv_bridge = CvBridge()

    def subscribe_listener(self):
        self.sub_color_image = rospy.Subscriber(
            self.color_topic, msg_Image, self.ImageColorCallback
        )
        self.sub_image = rospy.Subscriber(
            self.depth_topic, msg_Image, self.imageDepthCallback
        )

    def imageDepthCallback(self, data):
        print("image depth callback")

        np_arr = np.fromstring(data.data, np.uint16)
        np_arr = np.clip(np_arr, a_min=MIN_DEPTH, a_max=MAX_DEPTH)
        np_arr = (np_arr - MIN_DEPTH) / MAX_DEPTH
        # print(np.max(np_arr))
        image_np = np.reshape(np_arr, [480, 640, 1])
        image_np = pillow.fromarray(np.squeeze(image_np))
        # newsize = (128, 72)
        # image_np = image_np.resize(newsize)
        self.frames.append(image_np)
        self.curr_depth = np.asanyarray(image_np)

        imageio.imsave("img/depth_image_" + str(self.count) + ".png", image_np)
        self.count += 1

    def ImageColorCallback(self, data):
        print("image color callback")
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        self.rgb_img = self.rgb_img[:, :, ::-1]

        # np_arr = np.fromstring(data.data, np.uint8)
        # image_np = np.reshape(np_arr, [480, 640, 3])
        # image_np = pillow.fromarray(np.squeeze(image_np), mode='RGB')
        # # newsize = (128, 72)
        # # image_np = image_np.resize(newsize)
        # self.color_frames.append(image_np)
        # self.curr_color = np.asanyarray(image_np)
        rgb_img = pillow.fromarray(self.rgb_img, mode="RGB")
        # rgb_img.save('img/color_image_' + str(self.color_count) + '.jpg', "JPEG")

        # cv2.imwrite('img/color_image_' + str(self.color_count) + '.png', self.rgb_img)
        self.color_count += 1

    def poseCallback(self, data):
        poses = data.pose.pose
        # print(poses.position.x)
        self.poses.append(poses)
        self.curr_pose = poses


def collect_images():
    # depth_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_topic = "/camera/depth/image_rect_raw"
    color_topic = "/camera/color/image_raw"
    listener = ImagePoseListener(depth_topic, color_topic)
    listener.subscribe_listener()
    return listener


def save_image_data(listener, image_dir=None):
    if image_dir is None:
        image_dir = os.path.join(
            os.getcwd(), datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(image_dir)
    print(image_dir)
    poses_collected = listener.poses
    rpy_collected = listener.rpy
    np.save(os.path.join(image_dir, "poses"), poses_collected)
    np.save(os.path.join(image_dir, "rpy"), rpy_collected)
    print("listener frames: ", len(listener.frames))
    for i in range(len(listener.frames)):
        imageio.imsave(
            os.path.join(image_dir, "depth_image_" + listener.frames[i] + ".png")
        )
        print("saved depth image")
        if i < len(listener.color_frames):
            imageio.imsave(
                os.path.join(
                    image_dir, "color_image_" + listener.color_frames[i] + ".png"
                )
            )
            print("saved color image")

    # frames_collected = listener.frames
    # imageio.mimsave(os.path.join(image_dir, 'depth_image.mp4'), frames_collected, fps=5)
    # frames_collected = listener.color_frames
    # imageio.mimsave(os.path.join(image_dir, 'color_image.mp4'), frames_collected, fps=5)


if __name__ == "__main__":
    node_name = os.path.basename(sys.argv[0]).split(".")[0]
    print(node_name)
    rospy.init_node(node_name)
    collect_images()
    rospy.spin()
