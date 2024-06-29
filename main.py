from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

SIZE = 32
num_classes = 7

# Eğitimli modeli yükle
model = load_model(r"/Users/ibrahim/Desktop/workspace/FastAPI Practice/FastAPI-Skin-Lesion-Model/skin_lesion_model.h5")

# Label Encoder'ı yükle
le = LabelEncoder()
le.fit(['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])

# Template klasörünü ayarla
templates = Jinja2Templates(directory="templates")

# Statik dosyalar (CSS, JS) için klasörü ayarla
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Görüntü dosyasını oku
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((SIZE, SIZE))
        image = np.asarray(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Tahmin yap
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = le.inverse_transform(predicted_class)[0]

        # Tüm sınıflar için tahmin olasılıklarını al
        prediction_probabilities = prediction[0]

        response = {
            "predicted_label": predicted_label,
            "probabilities": {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(prediction_probabilities)}
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)