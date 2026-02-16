#!/usr/bin/env python3

import cv2
import json
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

# megapose
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.utils.logging import get_logger, set_logging_level
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

# megapose parameters
model_name = "megapose-1.0-RGB-multi-hypothesis"
cameraData = json.load(open("/home/logesh/6d_ws/megapose6d/local_data/extension_box/camera_data.json"))
camMat = cameraData['K']
camRes = cameraData['resolution']

logger = get_logger(__name__)

# object detection
objModel = YOLO("/home/logesh/6d_ws/megapose6d/local_data/detector/box_best.pt") 



# megapose helper functions

def load_observation_tensor(rgb_frame) -> ObservationTensor:
    depth = None
    observation = ObservationTensor.from_numpy(rgb_frame, depth, camMat)
    return observation

def load_detections(xyxy) -> DetectionsType:
    print(xyxy)
    infos = pd.DataFrame(
        dict(
            label=["box"],
            batch_im_id=0,
            instance_id=np.arange(1),
        )
    )
    bboxes = torch.as_tensor(
        np.stack([xyxy]),
    )
    detections = PandasTensorCollection(infos=infos, bboxes=bboxes).cuda()

    return detections

def make_object_dataset() -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    rigid_objects.append(RigidObject(label="box", mesh_path="/home/logesh/6d_ws/megapose6d/local_data/extension_box/meshes/box/box.ply", mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def estimate6DPose(frame, xyxy):
    """Simple function to estimate the pose the object in the frame, xyxy is the object bbox in the frame"""

    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(frame).cuda()
    detections = load_detections(xyxy).cuda()
    object_dataset = make_object_dataset()

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )

    return output

# main runner
while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (640, 480))

    # object detection (for bounding box)
    result = objModel.predict(source=frame, stream=False, conf=0.3, save=False)[0]
    cls = result.boxes.cls.cpu().tolist()
    box = result.boxes.xyxy.cpu().numpy()
    
    if (cls != []):
        # take only one bbox
        box = box[0]
        box = np.int64(box.reshape((2, 2)))
        # cv2.circle(frame, box[0], 5, (0,0,255), -1)
        # cv2.circle(frame, box[1], 5, (0,0,255), -1)

        # pose estimation
        output = estimate6DPose(frame, box.flatten())

        print(f"\n{output}\n")

    if (not _):
        print("No image read!")
        continue
    
    cv2.imshow("out", frame)
    key = cv2.waitKey(1)
    if (key == ord('q')):
        break

cv2.destroyAllWindows()
cam.release
