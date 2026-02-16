#!/usr/bin/env python3

# IMPORTANT for CUDA + Panda3D stability
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional
from scipy.spatial.transform import Rotation

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model


# -----------------------------
# Utility: Create object dataset
# -----------------------------
def make_object_dataset(mesh_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    for object_dir in mesh_dir.iterdir():
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}:
                mesh_path = fn
        assert mesh_path, f"No mesh found for {label}"
        rigid_objects.append(
            RigidObject(label=label, mesh_path=mesh_path, mesh_units="m")
        )
    return RigidObjectDataset(rigid_objects)


# -----------------------------
# Main MegaPose Direct Wrapper
# -----------------------------
class MegaPoseDirect:

    def __init__(
        self,
        model_name: str,
        mesh_dir: Path,
        K: np.ndarray,
        resolution: tuple,
        num_workers: int = 4,
    ):
        print("Loading object dataset...")
        self.object_dataset = make_object_dataset(mesh_dir)

        print("Loading model...")
        self.model_info = NAMED_MODELS[model_name]
        self.model = load_named_model(
            model_name,
            self.object_dataset,
            n_workers=num_workers,
        ).cuda()
        self.model.eval()

        print("Setting camera...")
        self.camera = CameraData()
        self.camera.K = K
        self.camera.resolution = resolution
        self.camera.z_near = 0.001
        self.camera.z_far = 10000

        print("Warmup...")
        self._warmup()

        print("MegaPose ready.\n")

    # -----------------------------
    def _warmup(self):
        h, w = self.camera.resolution
        dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        observation = ObservationTensor.from_numpy(
            dummy_img, None, self.camera.K
        ).cuda()

        labels = list(self.object_dataset.label_to_objects.keys())
        detections = self._make_detections(
            labels,
            np.asarray([[0, 0, w // 2, h // 2] for _ in labels])
        ).cuda()

        self.model.run_inference_pipeline(
            observation,
            detections=detections,
            **self.model_info["inference_parameters"]
        )

    # -----------------------------
    def _make_detections(self, labels: List[str], boxes: np.ndarray):
        objs = []
        for label, box in zip(labels, boxes):
            o = ObjectData(label)
            o.bbox_modal = box
            objs.append(o)
        return make_detections_from_object_data(objs)

    # -----------------------------
    def estimate_pose(
        self,
        image: np.ndarray,
        labels: List[str],
        boxes: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ):
        observation = ObservationTensor.from_numpy(
            image, depth, self.camera.K
        ).cuda()

        detections = self._make_detections(labels, boxes).cuda()

        output, extra = self.model.run_inference_pipeline(
            observation,
            detections=detections,
            **self.model_info["inference_parameters"]
        )

        return output, extra


# ===============================
# REAL-TIME LOOP
# ===============================
if __name__ == "__main__":

    mesh_dir = Path("/home/logesh/6d_ws/megapose6d/local_data/examples/box/meshes").absolute()

    K = np.asarray([
        [605.9547119140625, 0.0, 319.029052734375],
        [0.0, 605.006591796875, 249.67617797851562],
        [0, 0, 1]
    ])

    resolution = (480, 640)

    megapose = MegaPoseDirect(
        model_name="megapose-1.0-RGB-multi-hypothesis",
        mesh_dir=mesh_dir,
        K=K,
        resolution=resolution,
        num_workers=4,
    )

    cap = cv2.VideoCapture("/home/logesh/my_video-1.mkv")

    objModel = YOLO("/home/logesh/6d_ws/megapose6d/local_data/detector/box_best.pt") 

    """
    frame = cv2.imread("/home/logesh/6d_ws/megapose6d/local_data/examples/box/image_rgb0.png")
    frame = cv2.resize(frame, (640, 480))
    labels = ["box"]
    boxes = np.array([[100, 100, 500, 400]], dtype=np.float32)
    output, _ = megapose.estimate_pose(frame, labels, boxes)
    print("Pose:\n", output.poses)
    poses = output.poses.cpu()[0]
    tvec = np.array(poses[0:3, 3] / 1000.0)
    rvec = Rotation.from_matrix(poses[0:3, 0:3]).as_rotvec()
    cv2.drawFrameAxes(frame, K, np.zeros((5,)), rvec, tvec, 0.1, thickness=3)
    cv2.imshow("", frame)
    cv2.waitKey(0)
    """

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        # object detection (for bounding box)
        result = objModel.predict(source=frame, stream=False, conf=0.3, save=False)[0]
        cls = result.boxes.cls.cpu().tolist()
        box = result.boxes.xyxy.cpu().numpy()

        if (len(cls) > 0):
            labels = ["box"]
            box = np.float32(box[0].flatten())
            boxes = np.array([box], dtype=np.float32)

            output, _ = megapose.estimate_pose(frame, labels, boxes)

            poses = output.poses.cpu()[0]

            tvec = np.array(poses[0:3, 3] / 1000.0)
            rvec = Rotation.from_matrix(poses[0:3, 0:3]).as_rotvec()

            # cv2.drawFrameAxes(frame, K, np.zeros((5,)), rvec, tvec, 0.3, thickness=3)

            # cv2.rectangle(
            #     frame,
            #     (int(boxes[0][0]), int(boxes[0][1])),
            #     (int(boxes[0][2]), int(boxes[0][3])),
            #     (0, 255, 0),
            #     2,
            # )

        cv2.imshow("MegaPose Direct", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    # cap.release()
    cv2.destroyAllWindows()
    exit(0)