import typing as tp
import onnxruntime as ort
import numpy as np
import copy
from PIL import Image, ImageDraw
import streamlit as st


st.title('Object Detection App')
st.sidebar.write("## Control Panel:gear:")


def load_model(onnx_model_path: str = './model.onnx'):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 24
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_model_path, sess_options=sess_options)
    return ort_session


def predict(_ort_session, _img: Image.Image):

    img = np.array(_img).transpose((2, 0, 1))[np.newaxis, :, :, :]
    img = img.astype(np.float32) / 255.0

    ort_inputs = {_ort_session.get_inputs()[0].name: img}
    ort_outs = _ort_session.run(None, ort_inputs)
    return ort_outs


def draw(img: Image.Image, boxes: np.array, labels: tp.List[str]):
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        x0, y0 = boxes[i][0].item(), boxes[i][1].item()
        x1, y1 = boxes[i][2].item(), boxes[i][3].item()

        draw.rectangle(((x0, y0), (x1, y1)))
        draw.text((x1 - 15, y1 - 15), str(label))
    return img


ort_session = load_model()
categories = ['__background__', 'person', 'bicycle', 'car',
 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
 'suitcase', 'frisbee', 'skis', 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'N/A',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'N/A',
 'dining table',
 'N/A',
 'N/A',
 'toilet',
 'N/A',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'N/A',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']


col1, col2 = st.columns(2)
upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
box_threshold = st.sidebar.number_input("Box Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)


if upload:
    img = Image.open(upload)
    img = img.resize((512,512), Image.Resampling.BILINEAR)
    original_image = copy.deepcopy(img)
    boxes, labels, scores = predict(ort_session, img)

    indices = scores >= box_threshold
    boxes = boxes[indices]
    labels = labels[indices]
    labels = [categories[i.item()] for i in labels]

    img = draw(img, boxes, labels)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(np.array(original_image))
    with col2:
        st.header("Detections Image")
        st.image(np.array(img))
else:
    img = Image.open('coco_test.jpg')
    img = img.resize((512,512), Image.Resampling.BILINEAR)

    original_image = copy.deepcopy(img)
    boxes, labels, scores = predict(ort_session, img)

    indices = scores >= box_threshold
    boxes = boxes[indices]
    labels = labels[indices]
    labels = [categories[i.item()] for i in labels]

    img = draw(img, boxes, labels)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(np.array(original_image))
    with col2:
        st.header("Detections Image")
        st.image(np.array(img))
