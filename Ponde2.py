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
        
            Festivos = pd.DataFrame({
                 'holiday': 'Finde',
                 'ds': pd.to_datetime(['2019-11-01','2019-12-08','2019-12-24','2019-12-25','2019-12-31','2020-01-01','2020-04-10',
                     '2020-04-11','2020-05-01','2020-05-21','2020-06-29','2020-07-16','2020-08-15','2020-09-18','2020-09-19',
                     '2020-10-12','2020-10-25','2020-10-31','2020-11-01','2020-11-29','2020-12-08','2020-12-24','2020-12-25',
                     '2020-12-31','2021-01-01','2021-04-02','2021-04-03','2021-05-01','2021-05-21','2021-06-28','2021-07-16',
                     '2021-08-15','2021-09-17','2021-09-18','2021-09-19','2021-10-11','2021-10-31','2021-11-01','2021-12-08',
                     '2021-12-25','2022-01-01','2022-04-15','2022-04-16','2022-05-01','2022-05-21','2022-06-21','2022-06-27',
                     '2022-07-16','2022-08-15','2022-09-18','2022-09-19','2022-10-10','2022-10-31','2022-11-01','2022-12-08',
                     '2022-12-24','2022-12-25','2022-12-31','2021-01-02','2021-01-03','2021-01-09','2021-01-10','2021-01-16',
                     '2021-01-17','2021-01-23','2021-01-24','2021-01-30','2021-01-31','2021-02-06','2021-02-07','2021-02-13',
                     '2021-02-14','2021-02-20','2021-02-21','2021-02-27','2021-02-28','2021-03-06','2021-03-07','2021-03-13',
                     '2021-03-14','2021-03-20','2021-03-21','2021-03-27','2021-03-28','2021-04-03','2021-04-04','2021-04-10',
                     '2021-04-11','2021-04-17','2021-04-18','2021-04-24','2021-04-25','2021-05-01','2021-05-02','2021-05-08',
                     '2021-05-09','2021-05-15','2021-05-16','2021-05-22','2021-05-23','2021-05-29','2021-05-30','2021-06-05',
                     '2021-06-06','2021-06-12','2021-06-13','2021-06-19','2021-06-20','2021-06-26','2021-06-27','2021-07-03',
                     '2021-07-04','2021-07-10','2021-07-11','2021-07-17','2021-07-18','2021-07-24','2021-07-25','2021-07-31',
                     '2021-08-01','2021-08-07','2021-08-08','2021-08-14','2021-08-15','2021-08-21','2021-08-22','2021-08-28',
                     '2021-08-29','2021-09-04','2021-09-05','2021-09-11','2021-09-12','2021-09-18','2021-09-19','2021-09-25',
                     '2021-09-26','2021-10-02','2021-10-03','2021-10-09','2021-10-10','2021-10-16','2021-10-17','2021-10-23',
                     '2021-10-24','2021-10-30','2021-10-31','2021-11-06','2021-11-07','2021-11-13','2021-11-14','2021-11-20',
                     '2021-11-21','2021-11-27','2021-11-28','2021-12-04','2021-12-05','2021-12-11','2021-12-12','2021-12-18',
                     '2021-12-19','2021-12-25','2021-12-26','2022-01-01','2022-01-02','2022-01-08','2022-01-09','2022-01-15',
                     '2022-01-16','2022-01-22','2022-01-23','2022-01-29','2022-01-30','2022-02-05','2022-02-06','2022-02-12',
                     '2022-02-13','2022-02-19','2022-02-20','2022-02-26','2022-02-27','2022-03-05','2022-03-06','2022-03-12',
                     '2022-03-13','2022-03-19','2022-03-20','2022-03-26','2022-03-27','2022-04-02','2022-04-03','2022-04-09',
                     '2022-04-10','2022-04-16','2022-04-17','2022-04-23','2022-04-24','2022-04-30','2022-05-01','2022-05-07',
                     '2022-05-08','2022-05-14','2022-05-15','2022-05-21','2022-05-22','2022-05-28','2022-05-29','2022-06-04',
                     '2022-06-05','2022-06-11','2022-06-12','2022-06-18','2022-06-19','2022-06-25','2022-06-26','2022-07-02',
                     '2022-07-03','2022-07-09','2022-07-10','2022-07-16','2022-07-17','2022-07-23','2022-07-24','2022-07-30',
                     '2022-07-31','2022-08-06','2022-08-07','2022-08-13','2022-08-14','2022-08-20','2022-08-21','2022-08-27',
                     '2022-08-28','2022-09-03','2022-09-04','2022-09-10','2022-09-11','2022-09-17','2022-09-18','2022-09-24',
                     '2022-09-25','2022-10-01','2022-10-02','2022-10-08','2022-10-09','2022-10-15','2022-10-16','2022-10-22',
                     '2022-10-23','2022-10-29','2022-10-30','2022-11-05','2022-11-06','2022-11-12','2022-11-13','2022-11-19',
                     '2022-11-20','2022-11-26','2022-11-27','2022-12-03','2022-12-04','2022-12-10','2022-12-11','2022-12-17',
                     '2022-12-18','2022-12-24','2022-12-25','2022-12-31','2021-12-31','2021-12-24','2021-12-25','2021-12-30'])})
         
                               
            basico = Prophet(holidays=Festivos)
            basico.fit(df)
            futuro = basico.make_future_dataframe(periods=periods_input)
            forecast = basico.predict(futuro)
            Prediccion=  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input+1)
            Promedios=forecast[['ds', 'yhat','yhat_lower']]
            Salida = pd.DataFrame(data=Prediccion)
            dias=Salida['ds'].dt.strftime('%Y/%m/%d')
            Prediccion.to_excel("bandejas.xlsx", index=0)
            st.write(Salida)
            #st.write(Salida,index=False)
            #fig1=basico.plot(forecast)
            #st.write(fig1)
            fig = basico.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(), basico, forecast)
            st.write(fig)
            
            #st.checkbox('Mostrar componentes'):
            fig3 = basico.plot_components(forecast)
            st.write(fig3)
            #chart_data =pd.DataFrame(index=Salida['ds'], columns=['yhat', 'yhat_lower', 'yhat_upper'])

            #st.line_chart(chart_data)
            #st.line_chart(data=Salida,  columns=[['yhat', 'yhat_lower','yhat_upper']], use_container_width=True)
            
            
            #line_chart = alt.Chart(Salida).mark_line().encode(
            #x = 'ds:T',
            #y = "yhat:Q",tooltip=['ds:T', 'yhat']).properties(title="Grafico Evolutivo").interactive()
            #st.altair_chart(line_chart,use_container_width=True)
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
  
   
