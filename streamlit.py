import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image  # Remplacer cv2 par PIL
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

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
    # Convertir l'image en niveaux de gris avec Pillow
    drawn_image = Image.fromarray(drawn_image)
    drawn_image = drawn_image.convert('L')  # Convertir en niveaux de gris
    drawn_image = drawn_image.resize((28, 28))  # Redimensionner l'image
    drawn_image = np.array(drawn_image)  # Convertir en tableau numpy
    drawn_image = drawn_image.reshape(1, 28, 28, 1)  # Reshaper pour le modèle
    drawn_image = drawn_image / 255.0  # Normaliser
    prediction = model.predict(drawn_image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# Initialiser les variables de session
if 'validation_count_test' not in st.session_state:
    st.session_state.validation_count_test = 0
if 'correct_count_test' not in st.session_state:
    st.session_state.correct_count_test = 0
if 'validation_count_drawn' not in st.session_state:
    st.session_state.validation_count_drawn = 0
if 'correct_count_drawn' not in st.session_state:
    st.session_state.correct_count_drawn = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0  # To track canvas state for resetting
if 'last_drawn_prediction' not in st.session_state:
    st.session_state.last_drawn_prediction = None  # Store the last prediction results for drawn digits

# Fonction pour mettre à jour les statistiques de validation
def update_validation_stats():
    correct_test = st.session_state.correct_count_test
    total_test = st.session_state.validation_count_test
    correct_drawn = st.session_state.correct_count_drawn
    total_drawn = st.session_state.validation_count_drawn

    correct_total = correct_test + correct_drawn
    total_total = total_test + total_drawn

    if total_total > 0:
        correct_ratio = correct_total / total_total
        incorrect_ratio = 1 - correct_ratio
    else:
        correct_ratio = 0.0
        incorrect_ratio = 0.0

    return correct_ratio, incorrect_ratio

# Interface Streamlit
st.title("Reconnaissance de chiffres")

# Menu déroulant pour sélectionner une option
option = st.selectbox(
    "Choisissez une option",
    ("Donnée de Test", "Dessine moi un Chiffre", "Statistiques & Monitoring")
)

if option == "Donnée de Test":
    st.header("Chargement du fichier de test")

    # Utilisation de st.file_uploader pour charger un fichier CSV
    uploaded_file = st.file_uploader("Glissez-déposez votre fichier CSV ici", type=["csv"])

    if uploaded_file is not None:
        # Lecture du fichier CSV chargé
        X_test = pd.read_csv(uploaded_file).values
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_test = X_test / 255.0

        st.header("Affichage d'une image aléatoire du jeu de test")
        idx, image, predicted_digit, prediction = predict_random_image(X_test)
        st.image(image.squeeze(), width=150, caption=f"Indice: {idx}")
        st.write(f"Chiffre prédit: {predicted_digit}")

        # Affichage de la probabilité du chiffre prédit
        prob_predicted_digit = prediction[0][predicted_digit]
        st.write(f"Probabilité du chiffre prédit ({predicted_digit}): {prob_predicted_digit:.4f}")

        # Validation de la prédiction
        st.header("Valider la prédiction")

        with st.form(key='validation_form'):
            correct = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_test")
            correction = st.number_input("Entrez le chiffre correct (si non)", min_value=0, max_value=9, step=1, key="correction_input")
            submit_button = st.form_submit_button("Valider")

            if submit_button:
                st.session_state.validation_count_test += 1
                if correct == "Oui":
                    st.session_state.correct_count_test += 1
                    st.write("La prédiction a été validée comme correcte.")
                else:
                    st.write("La prédiction a été validée comme incorrecte.")
                    # Ajouter la prédiction à l'historique avec correction
                    st.session_state.predictions.append({
                        'source': 'Test',
                        'image_idx': idx,
                        'predicted_digit': predicted_digit,
                        'actual_digit': correction,
                        'probability': prob_predicted_digit,
                        'is_correct': False
                    })

                # Ajouter la prédiction à l'historique
                if correct == "Oui":
                    st.session_state.predictions.append({
                        'source': 'Test',
                        'image_idx': idx,
                        'predicted_digit': predicted_digit,
                        'probability': prob_predicted_digit,
                        'is_correct': True
                    })

        # Affichage des comptes pour les données de test
        st.write(f"Validation Count Test: {st.session_state.validation_count_test}")
        st.write(f"Correct Count Test: {st.session_state.correct_count_test}")

elif option == "Dessine moi un Chiffre":
    st.header("Dessiner un chiffre et prédire")

    # Réinitialiser le dessin si demandé
    if st.button("Réinitialiser le dessin"):
        st.session_state.canvas_key += 1
        st.session_state.last_drawn_prediction = None

    # Configuration du canvas avec des valeurs fixes
    canvas_result = st_canvas(
        stroke_width=9,
        stroke_color="#FFFFFF",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    # Bouton pour prédire le chiffre dessiné
    if st.button("Predict Drawn Digit"):
        if canvas_result.image_data is not None:
            drawn_image = canvas_result.image_data
            predicted_digit, prediction = predict_drawn_image(drawn_image)
            st.session_state.last_drawn_prediction = {
                'predicted_digit': predicted_digit,
                'probability': prediction[0][predicted_digit]
            }

    if st.session_state.last_drawn_prediction is not None:
        predicted_digit = st.session_state.last_drawn_prediction['predicted_digit']
        prob_predicted_digit = st.session_state.last_drawn_prediction['probability']
        st.image(canvas_result.image_data, width=150, caption=f"Chiffre dessiné")
        st.write(f"Chiffre prédit: {predicted_digit}")
        st.write(f"Probabilité du chiffre prédit: {prob_predicted_digit:.4f}")

        # Validation de la prédiction
        with st.form(key='validation_form_drawn'):
            correct = st.radio("La prédiction est-elle correcte?", ("Oui", "Non"), key="correct_drawn")
            correction = st.number_input("Entrez le chiffre correct (si non)", min_value=0, max_value=9, step=1, key="correction_input_drawn")
            submit_button = st.form_submit_button("Valider")

            if submit_button:
                st.session_state.validation_count_drawn += 1
                if correct == "Oui":
                    st.session_state.correct_count_drawn += 1
                    st.write("La prédiction a été validée comme correcte.")
                else:
                    st.write("La prédiction a été validée comme incorrecte.")
                    # Ajouter la prédiction à l'historique avec correction
                    st.session_state.predictions.append({
                        'source': 'Dessin',
                        'predicted_digit': predicted_digit,
                        'actual_digit': correction,
                        'probability': prob_predicted_digit,
                        'is_correct': False
                    })

                # Ajouter la prédiction à l'historique
                if correct == "Oui":
                    st.session_state.predictions.append({
                        'source': 'Dessin',
                        'predicted_digit': predicted_digit,
                        'probability': prob_predicted_digit,
                        'is_correct': True
                    })

        # Affichage des comptes pour les chiffres dessinés
        st.write(f"Validation Count Dessinés: {st.session_state.validation_count_drawn}")
        st.write(f"Correct Count Dessinés: {st.session_state.correct_count_drawn}")

elif option == "Statistiques & Monitoring":
    st.header("Statistiques")

    # Création des onglets
    tab1, tab2, tab3, tab4 = st.tabs(["Images de Test", "Chiffres Dessinés", "Globales", "Historique de prédictions"])

    with tab1:
        st.subheader("Statistiques des Images de Test")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Validations Test", st.session_state.validation_count_test)
            st.metric("Total Correct Test", st.session_state.correct_count_test)

        with col2:
            if st.session_state.validation_count_test > 0:
                test_success_rate = (st.session_state.correct_count_test / st.session_state.validation_count_test) * 100
            else:
                test_success_rate = 0.0
            st.metric("Pourcentage de Réussite Test", f"{test_success_rate:.2f}%")

    with tab2:
        st.subheader("Statistiques des Chiffres Dessinés")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Validations Dessinés", st.session_state.validation_count_drawn)
            st.metric("Total Correct Dessinés", st.session_state.correct_count_drawn)

        with col2:
            if st.session_state.validation_count_drawn > 0:
                drawn_success_rate = (st.session_state.correct_count_drawn / st.session_state.validation_count_drawn) * 100
            else:
                drawn_success_rate = 0.0
            st.metric("Pourcentage de Réussite Dessinés", f"{drawn_success_rate:.2f}%")

    with tab3:
        st.subheader("Statistiques Globales")

        # Calculer les KPI globaux
        total_validations = st.session_state.validation_count_test + st.session_state.validation_count_drawn
        total_correct = st.session_state.correct_count_test + st.session_state.correct_count_drawn

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Validations Globales", total_validations)
            st.metric("Total Correct Globales", total_correct)

        with col2:
            if total_validations > 0:
                overall_success_rate = (total_correct / total_validations) * 100
            else:
                overall_success_rate = 0.0
            st.metric("Pourcentage de Réussite Global", f"{overall_success_rate:.2f}%")

        # Calculer les proportions pour le graphique à secteurs
        correct_ratio, incorrect_ratio = update_validation_stats()

        # Création des graphiques interactifs
        st.subheader("Répartition des Prédictions Correctes et Incorrectes")

        # Graphique à secteurs
        # fig_pie = go.Figure(data=[go.Pie(
        #     labels=['Prédictions Correctes', 'Prédictions Incorrectes'],
        #     values=[correct_ratio * 100, incorrect_ratio * 100],
        #     hole=0.3,
        #     textinfo='label+percent',
        #     textfont_size=15
        # )])
        # fig_pie.update_layout(title_text="Proportion des Prédictions Correctes et Incorrectes", title_x=0.5)
        # st.plotly_chart(fig_pie, use_container_width=True)

        # Répartition des corrects et incorrects en fonction de la source
        correct_test = st.session_state.correct_count_test
        total_test = st.session_state.validation_count_test
        correct_drawn = st.session_state.correct_count_drawn
        total_drawn = st.session_state.validation_count_drawn

        percent_success_test = (correct_test / total_test) * 100 if total_test > 0 else 0.0
        percent_success_drawn = (correct_drawn / total_drawn) * 100 if total_drawn > 0 else 0.0

        # Calculer les pourcentages globaux
        percent_success_global = (total_correct / total_validations) * 100 if total_validations > 0 else 0.0

        fig_bar = go.Figure(data=[
            go.Bar(name='Correct', x=['Test', 'Dessin', 'Global'], y=[correct_test, correct_drawn, total_correct], marker_color='#1ABC9C'),
            go.Bar(name='Incorrect', x=['Test', 'Dessin', 'Global'], y=[total_test - correct_test, total_drawn - correct_drawn, total_validations - total_correct], marker_color='#2C3E50'),
            go.Bar(name='Pourcentage de Réussite', x=['Test', 'Dessin', 'Global'], y=[percent_success_test, percent_success_drawn, percent_success_global], marker_color='#F5B7B1', text=[f"{percent_success_test:.2f}%", f"{percent_success_drawn:.2f}%", f"{percent_success_global:.2f}%"], textposition='outside')
        ])
        fig_bar.update_layout(
            title_text='Répartition des Prédictions Correctes et Incorrectes',
            xaxis_title='Source',
            yaxis_title='Nombre',
            barmode='group',
            xaxis=dict(tickmode='array', tickvals=['Test', 'Dessin', 'Global'], ticktext=['Test', 'Dessin', 'Global']),
            yaxis=dict(tickformat='.2f')  # Format des valeurs de l'axe y
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Historique des Prédictions")
        if st.session_state.predictions:
            try:
                # Préparer les données pour le DataFrame
                history = []
                for pred in st.session_state.predictions:
                    entry = {
                        'Source': pred['source'],
                        'Prédiction': pred['predicted_digit'],
                        'Réel': str(pred.get('actual_digit', 'Correct')),  # Convertir en chaîne de caractères
                        'Probabilité': f"{pred['probability']:.2f}",  # Formater la probabilité avec 2 chiffres après la virgule
                        'Correct': 'Oui' if pred['is_correct'] else 'Non'
                    }
                    history.append(entry)

                # Créer le DataFrame sans les images brutes
                predictions_df = pd.DataFrame(history)

                # Appliquer le style
                def highlight_correct(val):
                    color = '#1ABC9C' if val == 'Oui' else '#F5B7B1'
                    return f'background-color: {color}'

                styled_df = predictions_df.style.applymap(highlight_correct, subset=['Correct'])

                # Afficher le DataFrame stylisé
                st.write(styled_df.to_html(), unsafe_allow_html=True)

            except Exception as e:
                st.write(f"Erreur lors de l'affichage des données : {e}")
        else:
            st.write("Aucune prédiction effectuée pour le moment.")
