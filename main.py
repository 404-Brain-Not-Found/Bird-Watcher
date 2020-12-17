import cv2
from process_image import rcnn_detection, draw_bounding_boxes
import time


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = vid.read()

        cv2.imshow("Video Capture", frame)

        birds = rcnn_detection(frame)
        frame = draw_bounding_boxes(frame, birds)

        cv2.imwrite(f"temp/{int(start_time)}.png", frame)
        print(f"Took: {time.time() - start_time} s")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()