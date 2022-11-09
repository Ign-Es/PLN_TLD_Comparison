import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.python.compiler.tensorrt import trt_convert_windows
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
LANE_COLORS = [(0,0,255),(0,255,0),(255,255,0),(0,255,255)]
TUSIMPLE_ROW_ANCHOR = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

class LaneDetector():
    def __init__(self, norm=True, input_shape=(288, 800, 3), output_shape=(56, 101, 4), model_path='models/PLN'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.norm = norm

    def detect_lanes(self, image):
        scaled_image, input_tensor = self.prepare_input(image)
        # Perform inference on the image
        y_pred = tf.squeeze(self.model.predict(input_tensor))
        image = scaled_image.numpy().astype(np.uint8)
        max_pred = tf.math.argmax(y_pred, 1)
        # Process output data
        for lane_num in range(4):
            lane = max_pred[:, lane_num].numpy() * self.input_shape[1]/(self.output_shape[1]-1)
            #print(lane)
            for lane_point_num in range(len(TUSIMPLE_ROW_ANCHOR)):
                cv2.circle(image, (lane[lane_point_num].astype(int), TUSIMPLE_ROW_ANCHOR[lane_point_num]), 2,
                           LANE_COLORS[lane_num], -1)
        return image

    def prepare_input(self, img):
        img = tf.image.resize(img, size=(self.input_shape[0], self.input_shape[1]))
        input_tensor = tf.identity(img)
        if self.norm:
            input_tensor = tf.image.per_image_standardization(img)
        input_tensor = input_tensor[np.newaxis, :, :, :]
        return img, input_tensor
