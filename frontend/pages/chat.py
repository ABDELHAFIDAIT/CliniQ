import streamlit as st
from utils.api_client import CliniQClient

def show():
    st.title("Assistant Médical CliniQ")
    st.info("Posez vos questions cliniques basées sur les protocoles de Polynésie Française.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: Quel est le traitement d'une diarrhée aiguë chez l'enfant ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyse des protocoles en cours..."):
                res = CliniQClient.ask_question(st.session_state.token, prompt)
                if res.status_code == 200:
                    data = res.json()
                    ans = data["answer"]
                    st.markdown(ans)
                    
                    # Affichage des métriques de confiance (DeepEval)
                    if data.get("metrics"):
                        with st.expander("Scores de fiabilité clinique"):
                            cols = st.columns(4)
                            metrics = data["metrics"]
                            cols[0].metric("Pertinence", f"{metrics.get('answerrelevancy', 0)*100:.0f}%")
                            cols[1].metric("Fidélité", f"{metrics.get('faithfulness', 0)*100:.0f}%")
                            cols[2].metric("Précision @k", f"{metrics.get('contextualprecision', 0)*100:.0f}%")
                            cols[3].metric("Rappel @k", f"{metrics.get('contextualrecall', 0)*100:.0f}%")

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error("Erreur lors de la génération de la réponse.")