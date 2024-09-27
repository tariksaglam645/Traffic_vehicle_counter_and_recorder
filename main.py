import numpy as np
import cv2 as cv
from ultralytics import YOLO
from sort.sort import *


class Traffic_Car_Controller:
    def __init__(self, video_path, model_path, line_cords):
        self.cap = cv.VideoCapture(video_path)
        self.model = YOLO(model_path)
        self.tracker = Sort()
        self.line_cords = line_cords
        self.line_counts = [0] * len(line_cords)
        self.line_cars = [[] for _ in range(len(line_cords))]
        self.setup_window()
        self.last_id = 0

    def setup_window(self):
        cv.namedWindow('Video', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('Video', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    def process_frame(self, frame):
        predictions = self.model(frame, stream=True)
        detections = []
        for r in predictions:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = map(int, box)
                detections.append([x1, y1, x2, y2, score])
        return np.asarray(detections)

    def update_lines(self, frame):
        for i, line in enumerate(self.line_cords):
            cv.line(frame, line[0], line[1], (255 * (i % 2), 0, 255 * (i % 3)), 4)

    def count_vehicles(self, frame, tracker_ids):
        for track in tracker_ids:
            x1, y1, x2, y2, track_id = map(int, track[:5])

            self.check_lines(frame, x1, y1, x2, y2, track_id)
            self.draw_tracker(frame, x1, y1, x2, y2, track_id)

    def check_lines(self, frame, x1, y1, x2, y2, track_id):

        for i, line in enumerate(self.line_cords):
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if line[0][0] - 10 <= center_x <= line[1][0] + 10 and line[0][1] - 10 <= center_y <= line[1][1] + 10:
                cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                if track_id not in self.line_cars[i]:
                    self.line_cars[i].append(track_id)
                    self.line_counts[i] += 1
                    self.save_car_img(frame, x1, y1, x2, y2, track_id)
                    self.last_id = track_id
        if self.last_id != 0:
            cv.putText(frame, f"car {self.last_id} successfully saved!", (20, 50),
                       fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 0),
                       thickness=2)

    def draw_tracker(self, frame, x1, y1, x2, y2, track_id):
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, str(track_id), (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def save_car_img(self, frame, x1, y1, x2, y2, track_id):
        cv.imwrite(f"Car_images/car_{track_id}.jpg", frame[y1:y2, x1:x2])

    def display_count(self, frame):
        for i, count in enumerate(self.line_counts):
            color = (255 * (i % 2), 0, 255 * (i % 3))
            cv.putText(frame, f"Line {i + 1} Car Count: {count}", (10, 400 + i * 40), cv.FONT_HERSHEY_SIMPLEX, 1, color,
                       2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.update_lines(frame)
            detections = self.process_frame(frame)
            tracker_ids = self.tracker.update(detections)
            self.count_vehicles(frame, tracker_ids)
            self.display_count(frame)

            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = "traffic.mp4"
    model_path = "best_cars.pt"
    line_cords = [[(472, 304), (739, 312)], [(780, 311), (1020, 308)], [(1351, 498), (1778, 718)]]
    counter = Traffic_Car_Controller(video_path, model_path, line_cords)
    counter.run()
