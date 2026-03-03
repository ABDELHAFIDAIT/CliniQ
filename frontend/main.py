import streamlit as st
from utils.api_client import CliniQClient

st.set_page_config(page_title="CliniQ - Assistant Médical", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None

def login_page():
    st.title("Connexion CliniQ")
    with st.form("login_form"):
        user = st.text_input("Nom d'utilisateur")
        pwd = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit:
            res = CliniQClient.login(user, pwd)
            if res.status_code == 200:
                st.session_state.token = res.json()["access_token"]
                st.rerun()
            else:
                st.error("Identifiants incorrects")

if not st.session_state.token:
    login_page()
else:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Assistant Chat", "Historique & Dashboard"])
    
    if st.sidebar.button("Déconnexion"):
        st.session_state.token = None
        st.rerun()

    if page == "Assistant Chat":
        import pages.chat as chat_view
        chat_view.show()
    else:
        import pages.history as history_view
        history_view.show()