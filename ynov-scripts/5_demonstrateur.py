# ---------------------
# Common imports
# ---------------------

import os

###### PAR DEFAULT ON DESACTIVE LES FONCTIONNALITES GPU !
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import time
import sys
import logging
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ynov import utils
from ynov.preprocessing import preprocess
from ynov.models_training import utils_models
# BUG: il faut import utils_deep_keras ici car pas géré par le cache de la fonction load_model
from ynov.models_training import utils_deep_keras

# Get logger
logger = logging.getLogger('ynov.5_demonstrateur')


# ---------------------
# Streamlit.io confs
# ---------------------

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError as e:
    logger.error("Impossible d'importer la librairie streamlit")
    logger.error("Veuillez installer streamlit sur votre virtual env : 'pip install streamlit' (testé avec 0.67.1)")
    logger.error("Il ne faut pas rajouter streamlit dans le install_requires du setup.py car rajoute trop de librairies en prod.")
    sys.exit("Can't import streamlit")

try:
    import altair as alt
except ImportError as e:
    logger.error("Impossible d'importer la librairie altair")
    logger.error("Veuillez installer altair sur votre virtual env : 'pip install altair' (testé avec 4.1.0)")
    logger.error("Il ne faut pas rajouter altair dans le install_requires du setup.py car rajoute trop de librairies en prod.")
    sys.exit("Can't import altair")

if not st._is_running_with_streamlit:
    logger.error('Ce script ne doit pas être exécuté par python, mais directement via streamlit')
    logger.error('e.g. "streamlit run 4_demonstrateur.py')
    sys.exit("Streamlit not started")

# ---------------------
# Streamlit.io SessionState
# ---------------------
# From https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
# Une session permet de garder des valeurs, même après un refresh

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get_session(**kwargs):
    """Gets a SessionState object for the current session"""
    # Hack to get the session object from Streamlit.
    ctx = ReportThread.get_report_ctx()
    this_session = None
    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError("Oh noes. Couldn't get your Streamlit Session object.")

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state


session = get_session(content=None)


# ---------------------
# Streamlit CSS update
# ---------------------

# On agrandit la taille du sidebar
css = '''
<style>
.sidebar.--collapsed .sidebar-content {
    margin-left: -30rem;
}
.sidebar .sidebar-content {
    width: 30rem;
}
code {
    display: block;
    white-space: pre-wrap;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


# ---------------------
# Fonctions utils
# ---------------------


@st.cache(allow_output_mutation=True)
def load_model(selected_model: str):
    '''Fonction pour load un model
    Utilisation du cache de streamlit.io

    Args:
        selected_model(str): nom du modèle à charger
    Returns:
        model (?): model chargé
        model_conf (dict): configuration du modèle
    '''
    model, model_conf = utils_models.load_model(selected_model)
    # On force un predict "bidon" pour initialiser le temps d'inférence
    # https://github.com/keras-team/keras/issues/8724
    tmp_df = pd.DataFrame({col:[0] for col in model.x_col})
    model.predict(tmp_df)
    return model, model_conf


@st.cache(allow_output_mutation=True)
def get_available_models():
    '''Fonction pour récupérer la liste des modèles disponibles

    Returns:
        list: liste des modèles disponibles
    '''
    # Start with an empty list
    models_list = []
    # Find models
    models_dir = utils.get_models_path()
    for path, subdirs, files in os.walk(models_dir):
        # On check la présence d'un .pkl (normalement condition suffisante)
        if len([f for f in files if f.endswith('.pkl')]) > 0:
            models_list.append(os.path.basename(path))
    models_list = sorted(models_list)
    return models_list


@st.cache(allow_output_mutation=True)
def get_model_conf_text(model_conf: dict):
    '''Fonction pour récupérer les informations à afficher pour un model donné

    Args:
        model_conf (dict): configuration d'un modèle
    Returns:
        str: texte markdown à afficher
    '''
    markdown_content = "---  \n"
    markdown_content += f"Tâche : {model_conf['model_type']}  \n"
    if 'multi_label' in model_conf.keys() and model_conf['multi_label'] is not None:
        markdown_content += f"Multi-label : {model_conf['multi_label']}  \n"
    # TODO : faire équivalent regression
    markdown_content += f"Date apprentissage : {model_conf['date']}  \n"
    markdown_content += f"Type modèle : {model_conf['model_name']}  \n"
    markdown_content += "---  \n"

    # Colonnes utiles
    markdown_content += "Colonnes utiles au modèle :\n"
    for col in model_conf['mandatory_columns']:
        markdown_content += f"- {col} \n"
    markdown_content += "---  \n"

    # Classifier
    if model_conf['model_type'] == 'classifier':
        markdown_content += "Labels du modèle : \n"
        if model_conf['multi_label']:
            for cl in model.list_classes:
                markdown_content += f"- {cl} \n"
        else:
            markdown_content += f"- {model_conf['y_col']} \n"
            for cl in model.list_classes:
                markdown_content += f"  - {cl} \n"
    # Regressor ? Rien à rajouter ?

    return markdown_content


def get_prediction(model, content: pd.DataFrame):
    '''Fonction pour obtenir les prédictions sur un contenu depuis un model donné

    Args:
        model (?): model à utiliser
        content (pd.DataFrame): contenu à traiter

    Returns:
        (?): predictions sur le contenu
        (?): probas sur le contenu
        (?): durée de la fonction
    '''
    start_time = time.time()

    # Preprocess
    if model.preprocess_pipeline is not None:
        df_prep = utils_models.apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()

    # Get predictions
    if model.model_type == 'classifier':
        predictions, probas = model.predict_with_proba(df_prep)
        # Get predictions with inverse transform
        predictions = predictions[0]
        probas = probas[0]
    else:
        predictions =  model.predict(df_prep)[0]
        probas = None

    # Return with prediction time
    prediction_time = time.time() - start_time
    return predictions, probas, prediction_time


def get_prediction_formatting_text(model, model_conf: dict, predictions, probas):
    '''Formatting des predictions en fonction du model (mono/multi, etc.)

    Args:
        model (?): model à utiliser
        model_conf (dict): configurations du model
        predictions (?): predictions à traiter
        probas (?): probabilités associées aux predictions

    Returns:
        (str): texte (en markdown) à afficher
    '''
    markdown_content = ''
    if model.model_type == 'classifier':
        if not model_conf['multi_label']:
            predictions_inversed = model.inverse_transform([predictions])[0]
            markdown_content = f"- {model_conf['y_col']}: **{predictions_inversed}**  \n"
            markdown_content += f"  - Probabilité : {round(probas[model.list_classes.index(predictions_inversed)] * 100, 2)} %  \n"
        else:
            # TODO : mettre un nombre de classes max ?
            markdown_content = ""
            for i, cl in enumerate(model.list_classes):
                if predictions[i] == 0:
                    markdown_content += f"- ~~{cl}~~  \n"
                else:
                    markdown_content += f"- **{cl}**  \n"
                markdown_content += f"  - Probabilité : {round(probas[i] * 100, 2)} %  \n"
    elif model.model_type == 'regressor':
        # TODO: gérer multioutput plus tard
        markdown_content = f"- {model_conf['y_col']}: **{predictions}**  \n"
    return markdown_content


@st.cache(allow_output_mutation=True)
def get_histogram(probas, list_classes: list, is_multi_label: bool):
    '''Fonction pour récupérer un histogramme des probabilités

    Args:
        probas (?): probabilités à traiter
        list_classes (list): classes du modèle traité
        is_multi_label (bool): si le model est mutli_label ou non
    Return:
        pd.DataFrame: dataframe avec les probas / classes
        (?): layer altair
    '''
    # Get dataframe
    if is_multi_label:
        predicted = ['Accepted' if proba >= 0.5 else 'Rejected' for proba in probas]
    else:
        max_proba = max(probas)
        predicted = ['Accepted' if proba == max_proba else 'Rejected' for proba in probas]
    df_probabilities = pd.DataFrame({'classes': list_classes, 'probabilités': probas, 'result': predicted})

    # Prepare plot
    domain = ['Accepted', 'Rejected']
    range_ = ['#1f77b4', '#d62728']
    bars = (
        alt.Chart(df_probabilities, width=720, height=80 * len(list_classes))
        .mark_bar()
        .encode(
            x=alt.X('probabilités:Q', scale=alt.Scale(domain=(0, 1))),
            y='classes:O',
            color=alt.Color('result', scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['probabilités:Q', 'classes:O'],
        )
    )
    # Nudges text to right so it doesn't appear on top of the bar
    text = bars.mark_text(align='left', baseline='middle', dx=3)\
               .encode(text=alt.Text('probabilités:Q', format='.2f'))

    return df_probabilities, alt.layer(bars + text)


# ---------------------
# Streamlit.io App
# ---------------------


st.title('Démonstrateur du projet ynov')
st.image(Image.open(os.path.join(os.path.dirname(utils.get_data_path()), 'ynov-ressources', 'nlp.jpg')), width=200)
st.markdown("---  \n")

# Gestion sidebar (sélection du modèle)
st.sidebar.title('Sélection du modèle')
selected_model = st.sidebar.selectbox('Choix du model', get_available_models(), index=0)


# Get model
if selected_model is not None:

    # ---------------------
    # Read the model
    # ---------------------

    # TODO: Possibilité plusieurs modèles ?
    start_time = time.time()
    model, model_conf = load_model(selected_model)
    model_loading_time = time.time() - start_time
    st.write(f"Model loading time: {round(model_loading_time, 2)}s (attention, peut être mis en cache par l'application)")
    st.markdown("---  \n")

    # ---------------------
    # Get model confs
    # ---------------------

    markdown_content = get_model_conf_text(model_conf)
    st.sidebar.markdown(markdown_content)

    # ---------------------
    # Inputs
    # ---------------------

    # TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO
    # TODO: à adapter à votre projet !!!
    # TODO: ici exemple sur des données "titanic"
    pid = st.text_input('(NON UTILE) PassengerId (NON UTILE)')
    pclass = st.select_slider('Pclass', options=[1, 2, 3], value=1)
    name = st.text_input('Name')
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', min_value=0, max_value=100, value=None)
    siblings = st.number_input('SibSp', min_value=0, max_value=10, value=0)
    parents_children = st.number_input('Parch', min_value=0, max_value=10, value=0)
    ticket = st.text_input('(NON UTILE) Ticket (NON UTILE)')
    fare = st.number_input('Fare', min_value=0, max_value=10000, value=0)
    ticket = st.text_input('(NON UTILE) Cabin (NON UTILE)')
    embarked = st.selectbox('Embarked', ['C', 'S', 'Q'])

    # Construct content from inputs
    content = pd.DataFrame({
        'PassengerId': [pid],
        'Pclass': [pclass],
        'Name': [name],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [siblings],
        'Parch': [parents_children],
        'Ticket': [ticket],
        'Fare': [fare],
        'Cabin': [ticket],
        'Embarked': [embarked],
    })
    # TODO TODO TODO TODO TODO
    # TODO TODO TODO TODO TODO

    # ---------------------
    # GO Button
    # ---------------------

    # Mise à jour des textes à prédire au clic sur le bouton prédire
    if st.button("Prédire"):
        session.content = content

    # On clear tout si appuie sur "Clear"
    if st.button("Clear"):
        session.content = None

    # Prédictions et affichage des résultats si données à traiter
    if session.content is not None:
        st.write("---  \n")
        st.markdown("## Résultats  \n")
        st.markdown("  \n")

        # ---------------------
        # Predictions
        # ---------------------

        predictions, probas, prediction_time = get_prediction(model, content)
        st.write(f"Prédictions (temps d'inférence : {int(round(prediction_time*1000, 0))}ms) :")

        # ---------------------
        # Formatage Predictions
        # ---------------------

        markdown_content = get_prediction_formatting_text(model, model_conf, predictions, probas)
        st.markdown(markdown_content)

        # ---------------------
        # Histogram probabilities
        # ---------------------

        if model.model_type == 'classifier':
            df_probabilities, altair_layer = get_histogram(probas, model.list_classes, model.multi_label)
            # Display dataframe probabilities & plot altair
            st.subheader('Histogramme des probabilités')
            st.write(df_probabilities)
            st.altair_chart(altair_layer)
        # TODO : faire quelque chose si regression ?

        st.write("---  \n")