import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

#df = pd.read_csv('fraudes_limpio.csv')

st.title("Probando Streamlit :hand:")
st.text("No Country Proyecto detección de fraudes")
st.image("fraude_imagen.png")

#st.radio("Elige una opción", df.columns)

#fraudes = df[df['isFraud'] == 1]

#fraudes_por_tipo = fraudes.groupby('type').size()
st.write("***Fraudes***")
#fig, ax = plt.subplots()
#fraudes_por_tipo.plot(kind='bar', ax=ax)
#ax.set_title('Fraudes por tipo de transacción')
#ax.set_xlabel('Tipo de transacción')
#ax.set_ylabel('Cantidad')
#st.pyplot(fig)

#st.dataframe(df.head())

#@st.cache