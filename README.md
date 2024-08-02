# Contexte du projet
Vous travaillez dans une startup edtech qui offre des services de vulgarisation de l'IA. Votre client souhaite avoir un outil de démo pour expliquer comment ça fonctionne un réseau de neurones.

Votre premier objectif est de créer un modèle de deep learning pour classifier des chiffres. Ensuite, vous devez développer l'interface utilisateur (avec streamlit) qui vous permet de déssiner un chiffre (ou choisir aléatoirement un numéro d'un dataset d'images) et de détecter s'il correspond à une chiffre entre 0 & 9.

# Interface utilisateur 

Pour l'interface utilisateur, votre première version streamlit doit être capable de montrer une image aléatoire du dataset test. Ensuite, grâce à un bouton "predict", votre modèle peut prédire le bon nombre et afficher le résultat dans l'interface. À la fin, l'interface doit vous permettre avec un bouton de valider si votre modèle a correctement classé l'image ou non.

Dans un deuxièmet temps, vous pouvez faire évoluer votre interface comme un jeu. Dans cette deuxième version, l'interface doit vous permettre de dessiner un nombre et le transformer en image. Cette image va ensuite être utilisée par le modèle pour prédire le bon nombre et afficher le résultat. L'idée derrière est de donner 10 opportunités à votre modèle de détecter le chiffre dessiné et donner des stats à la fin par rapport à sa performance de prédiction.

