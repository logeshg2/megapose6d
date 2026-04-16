## Steps to run examples scripts:

Currently my implementations of [inference.py](inference.py) and [custom_inference.py](custom_inference.py) will not work or will work badly (that is, bad pose estimation and low fps).

You can find the intput, outputs, mesh and other informations to run the example in [examples/box](local_data/examples/box).

### Official Working code - example

Object detection:

```bash
python -m megapose.scripts.run_inference_on_example box --vis-detections
```

Megapose-6D inference:

```bash
python -m megapose.scripts.run_inference_on_example box --run-inference
```

Visualizing 6D Pose of object:

```bash
python -m megapose.scripts.run_inference_on_example box --vis-outputs
```