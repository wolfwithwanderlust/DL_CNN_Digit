import streamlit as st
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Charger le modèle pré-entraîné
model = load_model('cnn_model.h5')

# Fonction pour afficher une image aléatoire et prédire le chiffre
def predict_random_image(X_test):
    idx = np.random.randint(0, X_test.shape[0])
    image = X_test[idx]
    image_reshaped = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image_reshaped)
    predicted_digit = np.argmax(prediction)
    return idx, image, predicted_digit, prediction

# Fonction pour prédire le chiffre dessiné
def predict_drawn_image(drawn_image):
    drawn_image = cv2.resize(drawn_image, (28, 28))
    drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2GRAY)
    drawn_image = drawn_image.reshape(1, 28, 28, 1)
    drawn_image = drawn_image / 255.0
    prediction = model.predict(drawn_image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# Lecture du jeu de test
test_dir = "test.csv"
X_test = pd.read_csv(test_dir).values
X_test = X_test.reshape(-1, 28, 28, 1)
X_test = X_test / 255.0

# Interface Streamlit
st.title("Reconnaissance de chiffres")

# Menu déroulant pour sélectionner une option
option = st.selectbox(
    "Choisissez une option",
    ("Donnée de Test", "Dessine moi un Chiffre", "Statistiques")
)

# Variables pour le suivi des validations
if 'validation_count_test' not in st.session_state:
    st.session_state.validation_count_test = 0
    st.session_state.correct_count_test = 0

if 'validation_count_drawn' not in st.session_state:
    st.session_state.validation_count_drawn = 0
    st.session_state.correct_count_drawn = 0
    st.session_state.predictions = []  # Liste pour stocker les prédictions

# Affichage selon l'option sélectionnée
if option == "Donnée de Test":
    st.header("Affichage d'une image aléatoire du jeu de test")
    idx, image, predicted_digit, prediction = predict_random_image(X_test)
    st.image(image.squeeze(), width=150, caption=f"Indice: {idx}")
    st.write(f"Chiffre prédit: {predicted_digit}")

    # Affichage de la probabilité du chiffre prédit
    prob_predicted_digit = prediction[0][predicted_digit]
    st.write(f"Probabilité du chiffre prédit ({predicted_digit}): {prob_predicted_digit:.4f}")

    # Validation de la prédiction
    st.header("Valider la prédiction")
    correct = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_test")
    if st.button("Valider", key="validate_test"):
        st.session_state.validation_count_test += 1
        if correct == "Oui":
            st.session_state.correct_count_test += 1
            st.write("La prédiction a été validée comme correcte.")
        else:
            st.write("La prédiction a été validée comme incorrecte.")
        # Debugging
        st.write(f"Validation Count Test: {st.session_state.validation_count_test}")
        st.write(f"Correct Count Test: {st.session_state.correct_count_test}")

elif option == "Dessine moi un Chiffre":
    st.header("Dessiner un chiffre et prédire")

    # Configuration du canvas avec des valeurs fixes
    canvas_result = st_canvas(
        stroke_width=9,  # Épaisseur du trait fixe
        stroke_color="#FFFFFF",  # Couleur du trait fixe (blanc)
        background_color="black",  # Fond noir
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Bouton pour prédire le chiffre dessiné
    if st.button("Predict Drawn Digit"):
        if canvas_result.image_data is not None:
            drawn_image = canvas_result.image_data

            # Prétraitement et prédiction de l'image dessinée
            predicted_digit, prediction = predict_drawn_image(drawn_image)

            # Afficher l'image prétraitée
            st.image(drawn_image.squeeze(), width=150, caption="Image dessinée après prétraitement")

            # Afficher la prédiction
            prob_predicted_digit = prediction[0][predicted_digit]
            st.write(f"Chiffre prédit: {predicted_digit}")
            st.write(f"Probabilité du chiffre prédit ({predicted_digit}): {prob_predicted_digit:.4f}")

            # Stocker la prédiction
            st.session_state.predictions.append({
                'predicted_digit': predicted_digit,
                'probability': prob_predicted_digit,
                'image': drawn_image,
                'is_correct': None  # Ajouter ce champ pour le statut de validation
            })

            # Validation de la prédiction
            correct_drawn = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_drawn")
            if st.button("Valider Drawn Digit", key="validate_drawn"):
                st.session_state.validation_count_drawn += 1
                if correct_drawn == "Oui":
                    st.session_state.correct_count_drawn += 1
                    st.session_state.predictions[-1]['is_correct'] = True
                    st.write("La prédiction du chiffre dessiné a été validée comme correcte.")
                else:
                    st.session_state.predictions[-1]['is_correct'] = False
                    st.write("La prédiction du chiffre dessiné a été validée comme incorrecte.")

                # Debugging status display
                st.write(f"Session State Predictions: {st.session_state.predictions}")

                # Réinitialiser le canvas pour dessiner une nouvelle image
                st.experimental_rerun()


    # Afficher l'historique des prédictions
    if st.session_state.predictions:
        st.subheader("Historique des Prédictions")
        for idx, pred in enumerate(st.session_state.predictions):
            st.image(pred['image'].squeeze(), width=150, caption=f"Prédiction {idx + 1}: {pred['predicted_digit']} ({pred['probability']:.4f})")
            validation_status = "Correcte" if pred['is_correct'] else "Incorrecte" if pred['is_correct'] is not None else "Non Validée"
            st.write(f"Statut: {validation_status}")
            # Debugging status display
            st.write(f"Debug - Predicted Digit: {pred['predicted_digit']}, Is Correct: {pred['is_correct']}")

elif option == "Statistiques":
    st.header("Statistiques de Validation")

    # Création des onglets pour les statistiques
    tab1, tab2 = st.tabs(["Statistiques Test", "Statistiques Dessins"])

    with tab1:
        st.subheader("Statistiques des Images de Test")
        if st.session_state.validation_count_test > 0:
            success_rate_test = (st.session_state.correct_count_test / st.session_state.validation_count_test) * 100
            failure_rate_test = 100 - success_rate_test
            st.write(f"Taux de réussite des images de test : {success_rate_test:.2f}%")
            st.write(f"Taux d'échec des images de test : {failure_rate_test:.2f}%")
        else:
            st.write("Aucune validation pour les images de test effectuée pour le moment.")

    with tab2:
        st.subheader("Statistiques des Chiffres Dessinés")
        if st.session_state.validation_count_drawn > 0:
            success_rate_drawn = (st.session_state.correct_count_drawn / st.session_state.validation_count_drawn) * 100
            failure_rate_drawn = 100 - success_rate_drawn
            st.write(f"Taux de réussite des chiffres dessinés : {success_rate_drawn:.2f}%")
            st.write(f"Taux d'échec des chiffres dessinés : {failure_rate_drawn:.2f}%")
        else:
            st.write("Aucune validation pour les chiffres dessinés effectuée pour le moment.")
