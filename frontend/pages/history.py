import streamlit as st
import pandas as pd
from utils.api_client import CliniQClient

def show():
    st.title("Dashboard & Historique")
    
    res = CliniQClient.get_history(st.session_state.token)
    if res.status_code == 200:
        history_data = res.json()
        if not history_data:
            st.write("Aucun historique disponible.")
            return

        df = pd.DataFrame(history_data)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%d/%m/%Y %H:%M')
        
        st.subheader("Vos dernières consultations")
        st.dataframe(df[['created_at', 'query', 'response']], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Détails des échanges")
        for item in history_data:
            with st.expander(f"{item['created_at']} - {item['query'][:50]}..."):
                st.write("**Question :**", item['query'])
                st.write("---")
                st.write("**Réponse CliniQ :**", item['response'])
    else:
        st.error("Impossible de charger l'historique.")