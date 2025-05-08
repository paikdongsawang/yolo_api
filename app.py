from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            name = model.names[cls_id]
            if name == "pothole":
                return JSONResponse({"result": "pothole"})
    return JSONResponse({"result": "safe"})
