import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

# Dictionary holding your global loaded model
ml_models = {}

# Must match the exact index order your model was trained on
CLASS_MAPPING = {
    0: {"species_id": "sp_monstera_del", "common": "Swiss Cheese Plant", "scientific": "Monstera deliciosa"},
    1: {"species_id": "sp_monstera_adan", "common": "Monkey Mask", "scientific": "Monstera adansonii"},
    2: {"species_id": "sp_snake_plant", "common": "Snake Plant", "scientific": "Dracaena trifasciata"},
}

CONFIDENCE_THRESHOLD = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_plant_model(weights_path: str = "ml_weights/plant_species_mobilenet.pth"):
    """Loads the CNN architecture and weights into memory ONCE at server startup."""
    num_classes = len(CLASS_MAPPING)

    # 1. Initialize the MobileNetV3 architecture
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # 2. Replace the final classification head to match your number of plant species
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    # 3. Load your custom trained weights (if the file exists)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded custom plant model weights from {weights_path}")
    else:
        print("WARNING: Custom .pth file not found. Running with base weights for testing!")

    model.to(DEVICE)
    model.eval()  # CRITICAL: Freeze dropout & batchnorm layers for inference
    return model


def predict_plant_species(image_bytes: bytes, model: nn.Module) -> dict:
    """Runs inference on raw image bytes and returns Top-3 candidates."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Use inference_mode() for faster execution and lower memory usage
    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Extract Top-k (up to 3, or fewer if you have fewer classes)
    k = min(3, len(CLASS_MAPPING))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    candidates = []
    for prob, idx in zip(top_probs, top_indices):
        plant_info = CLASS_MAPPING[int(idx)]
        candidates.append({
            "species_id": plant_info["species_id"],
            "common_name": plant_info["common"],
            "scientific_name": plant_info["scientific"],
            "confidence": round(float(prob), 4),
            "required_user_inputs": ["height_cm", "pot_size_inches"]
        })

    if candidates[0]["confidence"] >= CONFIDENCE_THRESHOLD:
        return {"status": "confirmed", "candidates": [candidates[0]]}
    else:
        return {"status": "ambiguous", "candidates": candidates}