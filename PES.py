import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np

import pystan
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json
import holidays

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
from datetime import datetime





st.set_page_config(page_title ="Forecast App",
                    initial_sidebar_state="collapsed",
                    page_icon= "🎁")


tabs = ["Aplicación","Creditos"]
page = st.sidebar.radio("Opciones",tabs)


@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)
def Subida():
    
    df_input = pd.DataFrame()  
    df_input=pd.read_excel(input)
    return df_input

def prep_data(df):

    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("La columna seleccionada se renombrara como: **ds** y los datos como : **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input


if page == "Aplicación":
    
         
        
    
    st.title('Predictor de Demanda 🧚‍♀️')
    st.write('Esta aplicacion esta hecha para poder predecir la demanda de bandejas para optimizar la planificacion de raciones, trabajo y compras.')
    
    df =  pd.DataFrame()   

    st.subheader('1. Subir Información 🏋️')
    st.write("Cargar Archivo Excel")
   
        
    input = st.file_uploader('')
    
    if input is None:
        st.write("Se debe cargar informacion en excel")
    

    try:
        if sample:
            st.markdown("")    
            
    except:

        if input:
            with st.spinner('Loading data..'):
                df = Subida()
        
                st.write("Columns:")
                st.write(list(df.columns))
                columns = list(df.columns)
        
                col1,col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Seleccionar la columna de Fecha",index= 0,options=columns,key="date")
                with col2:
                    metric_col = st.selectbox("Seleccionar la columna de valores",index=1,options=columns,key="values")

                df = prep_data(df)
                output = 0
    

        if st.checkbox('Mostrar Grafico',key='show'):
            with st.spinner('Plotting data..'):
                col1,col2 = st.columns(2)
                with col1:
                    st.dataframe(df)
                    
                with col2:    
                    st.write("Estadisticas:")
                    st.write(df.describe())

            try:
                line_chart = alt.Chart(df).mark_line().encode(
                    x = 'ds:T',
                    y = "y:Q",tooltip=['ds:T', 'y']).properties(title="Grafico Evolutivo").interactive()
                st.altair_chart(line_chart,use_container_width=True)
                
            except:
                st.line_chart(df['y'],use_container_width =True,height = 1000)
                
            
    st.subheader("2. Numero de días a Predecir 📆")

    with st.container():
        st.write('En esta Seccion se debe seleccionar la cantidad de dias que se desea predecir')
        st.write('Tener en cuenta que mientras mayor el horizonte de dias la precision puede disminuir')
            
        with st.expander("Horizonte de tiempo"):
            periods_input = st.number_input('Días a predecir',
            min_value = 1, max_value = 90,value=10)
                        
        
    if st.button('Procesar Prediccion',key='Aplica', ):
            
            basico = Prophet()
            basico.fit(df)
            futuro = basico.make_future_dataframe(periods=periods_input)
            forecast = basico.predict(futuro)
            Prediccion=  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input+1)
            Salida = pd.DataFrame(data=Prediccion) 
            dias=Salida['ds'].dt.strftime('%Y/%m/%d')
            Prediccion.to_excel("bandejas.xlsx", index=0)
            st.write(Salida)
            line_chart = alt.Chart(Salida).mark_line().encode(
            x = 'ds:T',
            y = "yhat:Q",tooltip=['ds:T', 'yhat']).properties(title="Grafico Evolutivo").interactive()
            st.altair_chart(line_chart,use_container_width=True)
            @st.cache
            def convert_df(Salida):
                 return Salida.to_csv(decimal=',',index=False).encode('utf-8')
            csv = convert_df(Salida)
     
            st.download_button(
              label="Descargar en csv",
              data=convert_df(Salida),
              file_name='Prediccion.csv',
              mime='text/csv',
                                )
          
    
if page == "Creditos":
    st.image("Esteban.png")   
