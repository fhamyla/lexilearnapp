from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Load the saved model, scaler, and label encoder
# -------------------------------
with open('best_dyslexia_focus_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('best_dyslexia_focus_model_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('best_dyslexia_focus_model_labels.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Dyslexia Focus Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or list your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request body schema
# -------------------------------
class StudentFeatures(BaseModel):
    age: float
    reading_speed: float
    reading_accuracy: float
    reading_comprehension: float
    writing_speed: float
    writing_quality: float
    grammar_sentence: float
    phonetic_spelling: float
    irregular_word_spelling: float
    spelling_accuracy: float

# -------------------------------
# Helper: Identify focus areas
# -------------------------------
def get_focus_areas(student_dict):
    """Return primary, secondary, tertiary focus areas based on lowest scores"""
    reading_avg = np.mean([student_dict['reading_speed'], student_dict['reading_accuracy'], student_dict['reading_comprehension']])
    writing_avg = np.mean([student_dict['writing_speed'], student_dict['writing_quality'], student_dict['grammar_sentence']])
    spelling_avg = np.mean([student_dict['phonetic_spelling'], student_dict['irregular_word_spelling'], student_dict['spelling_accuracy']])

    areas = {
        "Reading": reading_avg,
        "Writing": writing_avg,
        "Spelling": spelling_avg
    }

    # Sort by ascending score (weakest first)
    sorted_areas = sorted(areas.items(), key=lambda x: x[1])
    focus_areas = [
        {"name": sorted_areas[0][0], "score": round(sorted_areas[0][1], 2)},
        {"name": sorted_areas[1][0], "score": round(sorted_areas[1][1], 2)},
        {"name": sorted_areas[2][0], "score": round(sorted_areas[2][1], 2)}
    ]
    return focus_areas



@app.get("/")
def root():
    return {"message": "Backend is running!"}


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(student: StudentFeatures):
    # Convert input to numpy array
    features = np.array([[
        student.age,
        student.reading_speed,
        student.reading_accuracy,
        student.reading_comprehension,
        student.writing_speed,
        student.writing_quality,
        student.grammar_sentence,
        student.phonetic_spelling,
        student.irregular_word_spelling,
        student.spelling_accuracy
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict class
    pred_idx = model.predict(features_scaled)[0]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(label_encoder.classes_)}
    else:
        prob_dict = None

    # Identify focus areas
    student_dict = student.dict()
    focus_areas = get_focus_areas(student_dict)

    return {
        "predicted_difficulty": pred_class,
        "probabilities": prob_dict,
        "focus_areas": focus_areas
    }
