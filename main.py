import time
import os
import onnxruntime as ort
import numpy as np

from PIL import Image


# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# import torchvision.transforms as T
# import torchvision.transforms.functional as F
# from torchvision.utils import draw_bounding_boxes
# import numpy as np


import streamlit as st
from torch._C import iinfo

st.title('Object Detection App')
st.sidebar.write("## Control Panel:gear:")


def inference_onnx(onnx_model_path: str = './model.onnx'):
    # verify shape
    sess_options = ort.SessionOptions()

    img = np.array(Image.open('./000016.jpg'))
    img = img.transpose((2, 0, 1))[np.newaxis, :, :, :]
    img = img.astype(np.float32) / 255.0
    # Below is for optimizing performance
    sess_options.intra_op_num_threads = 24
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_model_path, sess_options=sess_options)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


outs = inference_onnx()

st.write(str(outs))

# weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# categories = weights.meta['categories']
# transforms = T.Compose([T.PILToTensor(), T.ConvertImageDtype(dtype=torch.float)])


# @st.cache_resource
# def load_model():
#     model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0)
#     model = model.eval()
#     return model


# # model = load_model()


# def predict(img, box_score_thresh: int):
#     img_tensor = transforms(img)
#     with torch.no_grad():
#         prediction = model(img_tensor.unsqueeze(0))[0]

#     indices = prediction['scores'] > box_score_thresh
#     prediction['boxes'] = prediction['boxes'][indices]
#     prediction['labels'] = prediction['labels'][indices]

#     prediction['labels'] = [categories[label.item()] for label in prediction['labels']]
#     return prediction


# col1, col2 = st.columns(2)
# upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# box_threshold = st.sidebar.number_input("Box Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# if upload:
#     img = Image.open(upload)
#     predictions = predict(img, box_score_thresh=box_threshold)
#     res = draw_bounding_boxes(F.pil_to_tensor(img).to(dtype=torch.uint8), boxes=predictions['boxes'], labels=predictions["labels"])
#     res = F.to_pil_image(res)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.header("Original Image")
#         st.image(np.array(img))
#     with col2:
#         st.header("Detections Image")
#         st.image(np.array(res))
