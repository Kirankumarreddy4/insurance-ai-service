from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")

version = project.version(1)

dataset = version.download("yolov8")