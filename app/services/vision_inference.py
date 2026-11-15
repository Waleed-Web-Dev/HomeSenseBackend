import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import json
import numpy as np
import onnxruntime as ort

# Dictionary holding your global loaded model
ml_models = {}
vision_state = {
    "species_session": None,
    "species_labels": {}
}


# Must match the exact index order your model was trained on
CLASS_MAPPING = {
    0: {"species_id": "sp_monstera_del", "common": "Swiss Cheese Plant", "scientific": "Monstera deliciosa"},
    1: {"species_id": "sp_monstera_adan", "common": "Monkey Mask", "scientific": "Monstera adansonii"},
    2: {"species_id": "sp_snake_plant", "common": "Snake Plant", "scientific": "Dracaena trifasciata"},
}

vision_state = {
    "species_session": None,
    "species_labels": {},
    "disease_session": None,
    "disease_labels": {},
}

# Actionable treatment plans mapped to the 6 universal disease classes
DISEASE_TREATMENTS = {
    "Healthy": {
        "severity": "none",
        "summary": "Leaves look vibrant with no visible pathogens or stress.",
        "treatment": "Continue standard watering and nutrition schedule."
    },
    "Fungal_Leaf_Spot": {
        "severity": "moderate",
        "summary": "Fungal leaf spots detected (likely due to excess leaf moisture or low airflow).",
        "treatment": "Prune heavily spotted leaves with sterile scissors, avoid misting leaves, and apply copper fungicide or diluted Neem oil every 5 days."
    },
    "Bacterial_Spot_Blight": {
        "severity": "high",
        "summary": "Water-soaked bacterial lesions or blight detected.",
        "treatment": "Isolate the plant immediately to prevent spreading. Remove infected foliage, water only at the soil base, and apply a copper-based bactericide."
    },
    "Leaf_Mold_Mildew": {
        "severity": "moderate",
        "summary": "Powdery mildew or foliar mold growth detected due to high humidity and stagnant air.",
        "treatment": "Wipe leaves gently with a damp cloth, increase room ventilation, and spray with a potassium bicarbonate or Neem oil solution."
    },
    "Spider_Mites_Pest": {
        "severity": "high",
        "summary": "Stippling and pest damage consistent with spider mites or sap-sucking insects.",
        "treatment": "Rinse foliage thoroughly, wipe undersides of leaves, and apply Neem oil or insecticidal soap every 3 days for two weeks."
    },
    "Yellowing_Nutrient_Stress": {
        "severity": "moderate",
        "summary": "Foliar chlorosis (yellowing) detected, indicating overwatering or nitrogen/iron deficiency.",
        "treatment": "Check top 2 inches of soil before watering again. If soil is dry and roots are healthy, apply half-strength balanced NPK fertilizer on the next cycle."
    }
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


def init_species_model(
        onnx_path: str = "ml_weights/species_mobilenetv3.onnx",
        labels_path: str = "ml_weights/species_labels.json"
):
    if not os.path.exists(onnx_path) or not os.path.exists(labels_path):
        print(f"WARNING: {onnx_path} not found yet. Running in fallback mode until placed in ml_weights/.")
        return

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    vision_state["species_session"] = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    with open(labels_path, "r") as f:
        vision_state["species_labels"] = json.load(f)
    print("Model 1 (Species ONNX) loaded successfully!")


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def run_species_inference(image_bytes: bytes) -> list[dict]:
    """Returns Top-3 [{species_id, confidence}] sorted highest to lowest."""
    session = vision_state["species_session"]
    id2label = vision_state["species_labels"]

    # Fallback if .onnx file hasn't been copied into ml_weights/ yet
    if session is None:
        return [
            {"species_id": "sp_calathea", "confidence": 0.52},
            {"species_id": "sp_calathea_lanc", "confidence": 0.34},
            {"species_id": "sp_peace_lily", "confidence": 0.14}
        ]

    # Preprocess image: Resize -> Normalize -> NCHW tensor
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    input_tensor = np.transpose(img_arr, (2, 0, 1))[np.newaxis, ...]

    # Run ONNX inference (~10ms)
    logits = session.run(["logits"], {"pixel_values": input_tensor})[0][0]
    probs = _softmax(logits)

    top_indices = np.argsort(probs)[::-1][:3]
    return [
        {"species_id": id2label[str(idx)], "confidence": round(float(probs[idx]), 4)}
        for idx in top_indices
    ]

def load_onnx_bundle(onnx_path: str, labels_path: str) -> dict:
    """Loads an ONNX Runtime session and its corresponding JSON label map."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    with open(labels_path, "r") as f:
        id2label = json.load(f)
    return {"session": session, "id2label": id2label}

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Preprocesses image for MobileNetV3 ONNX and computes canopy greenness ratio."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Calculate Canopy Greenness Ratio (G > R and G > B pixels)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.15)
    greenness_score = round(float(np.mean(green_mask)) * 100, 1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_array = (img_array - mean) / std
    tensor = np.transpose(norm_array, (2, 0, 1))[np.newaxis, ...]
    return tensor, greenness_score


def disease_inference(image_bytes: bytes, species_id_or_bundle=None) -> dict:
    """
    Runs ONNX inference for leaf disease detection using vision_state.
    Works whether passed (image_bytes, plant.species_id) or (image_bytes, bundle).
    """
    if isinstance(species_id_or_bundle, dict) and "session" in species_id_or_bundle:
        session = species_id_or_bundle["session"]
        id2label = species_id_or_bundle["id2label"]
    else:
        session = vision_state.get("disease_session")
        id2label = vision_state.get("disease_labels")
        if not session or not id2label:
            raise RuntimeError("Disease ONNX model is not loaded in vision_state.")

    input_tensor, greenness_score = preprocess_image(image_bytes)
    logits = session.run(["logits"], {"pixel_values": input_tensor})[0][0]
    probs = softmax(logits)

    top_idx = int(np.argmax(probs))
    confidence = round(float(probs[top_idx]), 4)
    raw_condition = id2label[str(top_idx)]

    is_healthy = raw_condition == "Healthy"
    guide = DISEASE_TREATMENTS.get(raw_condition, DISEASE_TREATMENTS["Healthy"])

    # Combine summary + treatment into 'advice' so /diagnose and Flutter receive a complete message
    combined_advice = f"{guide['summary']} {guide['treatment']}".strip()

    return {
        "healthy": is_healthy,
        "condition": raw_condition.replace("_", " "),
        "confidence": confidence,
        "advice": combined_advice,
        "canopy_greenness_pct": greenness_score,
        "severity": guide["severity"],
        "summary": guide["summary"],
        "treatment_plan": guide["treatment"],
    }

def init_disease_model(
    onnx_path: str = "ml_weights/disease_mobilenetv3.onnx",
    labels_path: str = "ml_weights/disease_labels.json",
):
    if not os.path.exists(onnx_path) or not os.path.exists(labels_path):
        print(f"WARNING: {onnx_path} not found yet. Disease checkups will run in fallback mode until placed in ml_weights/.")
        return

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    vision_state["disease_session"] = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    with open(labels_path, "r") as f:
        vision_state["disease_labels"] = json.load(f)
    print("Model 2 (Disease ONNX) loaded successfully!")