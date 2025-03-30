import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Désactiver les avertissements TensorFlow
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
MODEL_PATH = "src/models/finemail2.0.h5"
TOKENIZER_PATH = "src/models/tokenizerf.pkl"

# Gestion des erreurs pour le chargement du modèle
try:
    print("Chargement du modèle...")
    model = load_model(MODEL_PATH)
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    # Créer un modèle factice en cas d'erreur pour éviter le crash
    model = None

# Gestion des erreurs pour le chargement du tokenizer
try:
    print("Chargement du tokenizer...")
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    print("Tokenizer chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du tokenizer: {str(e)}")
    tokenizer = None

# Define a route for spam prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "Email text is required"}), 400
        
    # Vérifier si le modèle et le tokenizer sont chargés
    if model is None or tokenizer is None:
        return jsonify({"error": "Le modèle ou le tokenizer n'a pas pu être chargé"}), 500

    try:
        # Preprocess the email text
        sequences = tokenizer.texts_to_sequences([email_text])
        padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your model

        # Make a prediction
        prediction = model.predict(padded_sequences)
        is_spam = prediction[0][0] > 0.5  # Assuming binary classification (spam or not spam)

        return jsonify({
            "email": email_text,
            "is_spam": is_spam,
            "confidence": float(prediction[0][0])
        })
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)