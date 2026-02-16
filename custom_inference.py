#!/usr/bin/env python3

import cv2
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
# from ultralytics import YOLO
from torch.fx.experimental.optimization import optimize_for_inference, fuse

# megapose
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.utils.logging import get_logger, set_logging_level
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

# megapose parameters
model_name = "megapose-1.0-RGB-multi-hypothesis"
cameraData = CameraData.from_json((Path("/home/logesh/6d_ws/megapose6d/local_data/extension_box/camera_data.json")).read_text())

logger = get_logger(__name__)

# object detection
# objModel = YOLO("/home/logesh/6d_ws/megapose6d/local_data/detector/box_best.pt") 


# megapose helper functions

def load_observation_tensor(rgb_frame) -> ObservationTensor:
    depth = None
    rgb = np.array(rgb_frame, dtype=np.uint8)
    observation = ObservationTensor.from_numpy(rgb, depth, cameraData.K)
    return observation

def load_detections(xyxy) -> DetectionsType:
    obj = ObjectData("box")
    obj.bbox_modal = xyxy
    res = [obj]

    detections = make_detections_from_object_data(res).cuda()
    return detections

# def make_object_dataset() -> RigidObjectDataset:
#     rigid_objects = []
#     mesh_units = "mm"
#     rigid_objects.append(RigidObject(label="box", mesh_path="/home/logesh/6d_ws/megapose6d/local_data/extension_box/meshes/box/box.ply", mesh_units=mesh_units))
#     rigid_object_dataset = RigidObjectDataset(rigid_objects)
#     return rigid_object_dataset

def estimate6DPose(frame, xyxy):
    """Simple function to estimate the pose the object in the frame, xyxy is the object bbox in the frame"""

    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(frame).cuda()
    detections = load_detections(xyxy).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"], cuda_timer=True
    )

    return output

def make_object_dataset() -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (Path("/home/logesh/6d_ws/megapose6d/local_data/extension_box/") / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def main():
    # main runner
    import time
    img = cv2.imread("/home/logesh/6d_ws/megapose6d/local_data/examples/box/image_rgb.png")
    xyxy = [320, 240, 640, 480]
    t = time.perf_counter()
    output = estimate6DPose(img, xyxy)
    print("time taken:", time.perf_counter() - t)
    poses = output.poses.cpu().numpy()
    print(poses)

    return
    while True:
        _, frame = cam.read()

        if (not _):
            print("No image read!")
            continue

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
        
        cv2.imshow("out", frame)
        key = cv2.waitKey(1)
        if (key == ord('q')):
            break


if __name__ == "__main__":
    # class Optimized(torch.nn.Module):
    #     def __init__(self, m: torch.nn.Module, inp):
    #         super().__init__()
    #         self.m = m.eval()
    #         self.m = fuse(self.m, inplace=False)
    #         self.m = torch.jit.trace(self.m, torch.rand(inp).cuda())
    #         self.m = torch.jit.freeze(self.m)

    #     def forward(self, x):
    #         return self.m(x).float()

    # h, w = (480, 640)
    # pose_estimator.coarse_model.backbone = Optimized(pose_estimator.coarse_model.backbone, (1, 9, h, w))
    # pose_estimator.refiner_model.backbone = Optimized(pose_estimator.refiner_model.backbone, (1, 32 if False else 27, h, w))
    set_logging_level("info")
    
    object_dataset = make_object_dataset()

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    pose_estimator.eval()
    
    main()

    # cv2.destroyAllWindows()
    # cam.release
