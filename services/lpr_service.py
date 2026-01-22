import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate

class LicensePlateService:
    def __init__(seft):
        seft.vehicle_model = YOLO("models/yolov8n.pt")
        seft.plate_model = YOLO("models/license_plate_detector.pt")
        seft.tracker = Sort()

        seft.vehicles = [2,3,5,7]


    def process_image(self, frame: np.ndarray):
        results = []
        vehicle_dets = self.vehicle_model(frame)[0]
        detections = []

        for box in vehicle_dets.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in self.vehicles:
                detections.append([x1, y1, x2, y2, score])

        track_ids = self.tracker.update(np.array(detections)) if detections else []

        plates = self.plate_model(frame)[0]
        print("Number of plates:", len(plates.boxes))

        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = plate

            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

            text, text_score = read_license_plate(crop)

            print(f"plate : {text}")

            if text:
                results.append({
                    "license_plate": text,
                    "text_score": float(text_score),
                    "plate_bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return results