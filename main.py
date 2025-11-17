import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

app = FastAPI()

class_names = ["colgate", "vim", "cola"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)
        return {"prediction": class_names[pred.item()]}
@app.get("/")
def root():
    return {"message": "Model API is running!"}
