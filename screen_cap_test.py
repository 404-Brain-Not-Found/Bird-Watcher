from process_image import rcnn_detection, draw_bounding_boxes
from PIL import ImageGrab
import numpy as np
import cv2

if __name__ == "__main__":
    while 1:
        img = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = rcnn_detection(img, 0.95)
        img = draw_bounding_boxes(img, boxes)
        cv2.imshow("Processed Image", img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
