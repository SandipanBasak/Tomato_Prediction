from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io

# -------------------- APP CONFIG --------------------
app = FastAPI(title="Tomato Disease Detection API", version="1.0")

MODEL_PATH = "tomato_resnet50_model_full.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- CLASS NAMES --------------------
CLASS_NAMES = {
    0: "Tomato_Bacterial_spot",
    1: "Tomato_Early_blight",
    2: "Tomato_Late_blight",
    3: "Tomato_Leaf_Mold",
    4: "Tomato_Septoria_leaf_spot",
    5: "Tomato_Spider_mites_Two_spotted_spider_mite",
    6: "Tomato_Target_Spot",
    7: "Tomato_Yellow_Leaf_Curl_Virus",
    8: "Tomato_Mosaic_Virus",
    9: "Tomato_Healthy"
}

# -------------------- CURE & PREVENTION DATA --------------------
DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "cure_map": [
            "Spray Copper-based bactericides like Copper Oxychloride (0.3%) or Copper Hydroxide.",
            "Remove and destroy infected leaves and fruits immediately.",
            "Avoid working in wet fields to prevent spread through water droplets."
        ],
        "prevention_map": [
            "Use certified disease-free seeds and resistant varieties.",
            "Avoid overhead irrigation to reduce leaf wetness.",
            "Rotate crops with non-solanaceous plants for at least 2–3 years."
        ]
    },
    "Tomato_Early_blight": {
        "cure_map": [
            "Spray Mancozeb (0.25%) or Chlorothalonil at early infection.",
            "Remove infected leaves and apply Carbendazim or Azoxystrobin alternately.",
            "Avoid overhead irrigation and ensure good field aeration."
        ],
        "prevention_map": [
            "Use disease-free seeds and resistant varieties.",
            "Mulch soil to prevent soil-borne spores from splashing onto leaves.",
            "Follow proper crop rotation and avoid water stress."
        ]
    },
    "Tomato_Late_blight": {
        "cure_map": [
            "Spray Metalaxyl (0.2%) or Mancozeb (0.25%) at early stages of disease.",
            "Remove infected plant parts and avoid reusing contaminated soil.",
            "Repeat fungicide application every 10–12 days during wet periods."
        ],
        "prevention_map": [
            "Use resistant varieties and certified seeds.",
            "Avoid dense planting to promote air circulation.",
            "Monitor regularly during humid weather and apply preventive sprays."
        ]
    },
    "Tomato_Leaf_Mold": {
        "cure_map": [
            "Apply fungicides like Chlorothalonil or Copper Oxychloride (0.3%) as soon as symptoms appear.",
            "Remove and destroy infected leaves.",
            "Maintain moderate humidity in greenhouses."
        ],
        "prevention_map": [
            "Ensure proper ventilation and spacing between plants.",
            "Avoid frequent overhead irrigation.",
            "Disinfect tools and greenhouse structures regularly."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "cure_map": [
            "Spray Mancozeb (0.25%) or Copper fungicides weekly during humid conditions.",
            "Remove infected leaves to reduce spore load.",
            "Ensure the soil is well-drained and not waterlogged."
        ],
        "prevention_map": [
            "Use disease-free seeds and rotate crops with cereals or legumes.",
            "Avoid wetting foliage during irrigation.",
            "Maintain field hygiene and remove plant debris."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "cure_map": [
            "Spray Neem oil (2%) or Abamectin (0.002%) on affected leaves.",
            "Maintain adequate moisture to reduce mite populations.",
            "Use biological control agents like predatory mites."
        ],
        "prevention_map": [
            "Avoid drought stress by maintaining uniform irrigation.",
            "Regularly inspect the underside of leaves for early signs of mites.",
            "Avoid excessive use of insecticides that kill natural predators."
        ]
    },
    "Tomato_Target_Spot": {
        "cure_map": [
            "Spray fungicides like Chlorothalonil or Azoxystrobin when symptoms first appear.",
            "Remove and destroy infected leaves and crop debris.",
            "Avoid overhead irrigation to reduce leaf wetness."
        ],
        "prevention_map": [
            "Use resistant varieties and practice crop rotation.",
            "Ensure good field sanitation and weed control.",
            "Improve air circulation by proper plant spacing."
        ]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "cure_map": [
            "Remove and destroy infected plants immediately.",
            "Control whiteflies using Imidacloprid (0.005%) or Neem oil sprays.",
            "Apply reflective mulches to repel vector insects."
        ],
        "prevention_map": [
            "Use virus-free seedlings and resistant varieties.",
            "Install yellow sticky traps to monitor whiteflies.",
            "Avoid planting near other solanaceous crops with active whitefly infestations."
        ]
    },
    "Tomato_Mosaic_Virus": {
        "cure_map": [
            "Remove and destroy infected plants and nearby weeds.",
            "Disinfect tools and equipment with a 1% sodium hypochlorite solution.",
            "Avoid tobacco use while handling tomato plants to prevent virus spread."
        ],
        "prevention_map": [
            "Use virus-resistant varieties.",
            "Wash hands and tools before handling plants.",
            "Avoid mechanical injury to leaves which facilitates virus entry."
        ]
    },
    "Tomato_Healthy": {
        "cure_map": ["No treatment required — the plant is healthy."],
        "prevention_map": [
            "Maintain proper fertilization and irrigation schedules.",
            "Regularly scout for pests and early symptoms of disease.",
            "Follow crop rotation and good agricultural practices."
        ]
    }
}

# -------------------- LAZY MODEL LOADING --------------------
model = None  # global model variable

def load_model():
    global model
    if model is None:
        print("🔄 Loading model into memory...")
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        model.to(device)
        print("✅ Model loaded successfully!")
    return model

# -------------------- IMAGE PREPROCESSING --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- ROUTES --------------------
@app.get("/")
def home():
    return {"message": "✅ Tomato Disease Detection API is running and ready to accept images!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lazy load model only when needed
        model_instance = load_model()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_instance(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[preds.item()]

        disease_info = DISEASE_INFO.get(predicted_class, {"cure_map": [], "prevention_map": []})

        return JSONResponse({
            "prediction": predicted_class,
            "cure_recommendations": disease_info["cure_map"],
            "prevention_tips": disease_info["prevention_map"]
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
