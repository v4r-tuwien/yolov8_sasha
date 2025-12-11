from ultralytics import YOLO
import rospy
from actionlib import SimpleActionServer
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import ros_numpy

import numpy as np

import argparse
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

names_mapping = {
    "001_chips_can": "obj_000001",
    "002_master_chef_can": "obj_000002",
    "003_cracker_box": "obj_000003",
    "004_sugar_box": "obj_000004",
    "005_tomato_soup_can": "obj_000005",
    "006_mustard_bottle": "obj_000006",
    "008_pudding_box": "obj_000007",
    "009_gelatin_box": "obj_000008",
    "010_potted_meat_can": "obj_000009",
    "011_banana": "obj_000010",
    "013_apple": "obj_000011",
    "014_lemon": "obj_000012",
    "015_peach": "obj_000013",
    "016_pear": "obj_000014",
    "017_orange": "obj_000015",
    "018_plum": "obj_000016",
    "021_bleach_cleanser": "obj_000017",
    "024_bowl": "obj_000018",
    "025_mug": "obj_000019",
    "029_plate": "obj_000020"
}

class YOLOv8:
    def __init__(
            self,
            weights='yolov8s.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/fleckerl.yaml',  # dataset.yaml path
            imgsz=(480, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            camera_topic='/camera/color/image_raw',
                ):

        self.img_size = imgsz
        self.conf_thres= conf_thres
        self.iou_thres = iou_thres
        self.camera_topic = camera_topic
        self.device = device

        self.model = YOLO(weights)  # load a custom model
        self.is_ycb_ichores = ("ichores" in str(weights))

        print("\n\n\n")
        print(weights, device, data, camera_topic)
        print("\n\n\n")

        # ROS Stuff
        self.bridge = CvBridge()
        self.server = SimpleActionServer('/object_detector/yolov8', GenericImgProcAnnotatorAction, self.service_call, False)

        self.server.start()
        print("Server started, waiting for requests...")

    def service_call(self, goal):
        rgb = goal.rgb
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            img0 = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        ros_detections = self.infer(img0, rgb.header) 

        if ros_detections.success:
            self.server.set_succeeded(ros_detections)
        else:
            self.server.set_aborted(ros_detections)

    def infer(self, im0s, rgb_header):
        height, width, channels = im0s.shape

        results = self.model(im0s, conf=self.conf_thres, iou=self.iou_thres, device=self.device, verbose=False)  # predict on an image
        detections = []

        cls = results[0].boxes.cls.cpu().detach().numpy()

        if len(cls):
            boxes = results[0].boxes.xyxy.cpu().detach().numpy()
            masks = results[0].masks.data.cpu().detach().numpy()

            conf = results[0].boxes.conf.cpu().detach().numpy()
            names = results[0].names
            bboxes = []
            class_names = []
            confidences = []
            label_image = np.full((height, width), -1, np.int16)
            for idx in range(len(cls)):
                if self.is_ycb_ichores:
                    class_names.append(names_mapping[names[cls[idx]]])
                else:
                    class_names.append(names[cls[idx]])
                
                bb = RegionOfInterest()
                xmin = int(boxes[idx][0])
                ymin = int(boxes[idx][1])
                xmax = int(boxes[idx][2])
                ymax = int(boxes[idx][3])
                bb.x_offset = xmin
                bb.y_offset = ymin
                bb.height = ymax - ymin
                bb.width = xmax - xmin
                bb.do_rectify = False
                bboxes.append(bb)
                # ---
                # mask
                # ---
                mask = masks[idx]
                
                label_image[mask > 0] = idx
                # ---
                confidences.append(conf[idx])
                # ---
                # 
            mask_image = ros_numpy.msgify(Image, label_image, encoding='16SC1')
    
            server_result = GenericImgProcAnnotatorResult()
            server_result.success = True
            server_result.bounding_boxes = bboxes
            server_result.class_names = class_names
            server_result.class_confidences = confidences
            server_result.image = mask_image
        else:
            server_result = GenericImgProcAnnotatorResult()
            server_result.success = False

        return server_result    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/segment/train10/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--camera-topic', type=str, default='/camera/color/image_raw', help='camera topic for input image')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

# python detect_ros_seg.py --weights ./runs/segment/train10/weights/best.pt --conf-thres 0.9 --camera-topic '$$(rosparam get /pose_estimator/color_topic)'
if __name__ == "__main__":

    try:
        rospy.init_node('yolov8')
        opt = parse_opt()
        YOLOv8(**vars(opt))
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

