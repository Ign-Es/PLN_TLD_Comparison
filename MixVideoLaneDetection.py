import cv2
from src.LaneDetector import LaneDetector
from src import tld
import numpy as np


video_path = "Videos/testing_video2.avi"
model_path = 'trained_pln_tusimple'
# Initialize video
cap = cv2.VideoCapture(video_path)

#videoUrl = 'https://youtu.be/2CIxM7x-Clc'
#videoPafy = pafy.new(videoUrl)
#print(videoPafy.streams)
#cap = cv2.VideoCapture(videoPafy.streams[-1].url)

# Set hsv ranges for colors
yellow_range = {'low': [25, 140, 100], 'high': [45, 255, 255]}
white_range = {'low': [0, 0, 150], 'high': [180, 100, 255]}
red_range = {'low_1': [0,140,100], 'high_1': [15,255,255], 'low_2': [165,140,100], 'high_2': [180,255,255]}
colors = {"red": red_range, "yellow": yellow_range, "white": white_range}
color_ranges = {color: tld.ColorRange.fromDict(d) for color, d in list(colors.items())}

# Initialize deep learning lane detection model
lane_detector = LaneDetector(norm=True, model_path=model_path)
# Initialize traditional detector
detector = tld.LineDetector()

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
    except:
        continue

    if ret:

        # Detect the lanes
        pln_img = lane_detector.detect_lanes(frame)
        # Resize the image to the desired dimensions
        height_original, width_original = frame.shape[0:2]
        img_size = (800, 288)
        top_cutoff = int(height_original * 0.)
        if img_size[0] != width_original or img_size[1] != height_original:
            frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_NEAREST)
        frame = frame[top_cutoff:, :, :]

        # Extract the line segments for every color
        detector.setImage(frame)
        detections = {
            color: detector.detectLines(ranges) for color, ranges in list(color_ranges.items())
        }

        # Remove the offset in coordinates coming from the removing of the top part and
        arr_cutoff = np.array([0, top_cutoff, 0, top_cutoff])
        arr_ratio = np.array(
            [
                1.0 / img_size[1],
                1.0 / img_size[0],
                1.0 / img_size[1],
                1.0 / img_size[0],
            ]
        )
        # Plot Segments
        colorrange_detections = {color_ranges[c]: det for c, det in list(detections.items())}
        tld_img = tld.plotSegments(frame, colorrange_detections)
        im_mix = cv2.vconcat([pln_img, tld_img])
        cv2.imshow("Detected lanes", im_mix)
        #cv2.waitKey(0)
    else:
        break

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()