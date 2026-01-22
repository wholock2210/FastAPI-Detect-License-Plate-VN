from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from services.lpr_service import LicensePlateService

app = FastAPI(
    title="Vietnam License Plate Recognition API",
    version="1.0.0"
)

lpr_service = LicensePlateService()


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "success": False,
            "message": "Invalid image"
        }

    results = lpr_service.process_image(img)

    return {
        "success": True,
        "count": len(results),
        "results": results
    }
