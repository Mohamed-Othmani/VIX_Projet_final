
import pandas as pd
import streamlit as st
import yfinance as yf
from PIL import Image
import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
import numpy as np
from datetime import date, timedelta
import pmdarima as pm
from pmdarima.arima import auto_arima 
import pickle
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA




#main indique que l'application commence ici
if __name__ == "__main__":

    ####### PAGE CONFIG ###########################
    st.set_page_config(
    page_title="Analyse du Vix",
    page_icon="📈",
    layout = "centered",
    initial_sidebar_state="expanded",
    )





st.title('Analyse du VIX 📈')



st.markdown("""
            Bienvenue sur notre première page d'analyse d'indice financier. 
            Le premier indice que nous présenterons ici sera l'indice de la peur et de la cupidité : le VIX ! 
            Cet indice contrôle un autre indice qui est la volatilité. 
            La volatilité mesure l'ampleur et la rapidité de l'évolution du prix d'un actif sur une période donnée.
            Ici, nous parlerons pour le moment uniquement du VIX.
    
""")

st.markdown(""" Voici une courte vidéo explicative de ce qu'est le VIX produite par le site officiel de la Bourse de Chicago.""")

video_file = open('Understanding the VIX.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes) 



st.markdown("""Afin d'analyser et de produire de bonnes prédictions, nous avons besoin de beaucoup de données temporelles.
            Nous avons trouvé pour ce projet un ensemble de données intéressant qui retrace l'évolution du VIX depuis plusieurs années.""")


st.write("""si vous souhaitez voir le dataset de plus prés cliquer sur la petite box""")

@st.cache
def csv(nrows):
    df =pd.read_csv("dataset_VIX.csv", nrows = nrows)

    return df

df = csv(500)

if st.checkbox('Show raw VIX data'):
   st.subheader('Raw data')
   st.write(df)
#figg = px.histogram(df, x="Date", y = "Close")
#st.plotly_chart(figg)


st.area_chart(df["Close"])

#fig =px.bar(df["Close"])
#st.bar_chart(fig)
st.header("Le SP500")
st.markdown("""
    Le S&P 500 est un indice boursier basé sur 500 grandes sociétés cotées sur les bourses aux États-Unis. 
    Il couvre environ 80 % du marché boursier américain, c'est sur certaine options de ce marché que le VIX se base.
""")

def sp_data(nrows):
    data =pd.read_csv("s&p500.csv", nrows = nrows)
    return data

sp = sp_data(100)

if st.checkbox('Show raw SP data'):
    st.subheader('Raw data.')
    st.write(sp)
    
    
st.header("Bonus video ✅")    
st.markdown("""Si vous souhaitez en apprendre plus et pousser votre compréhension, voici quelques vidéos intéressantes pour vous.""")

with st.expander("⏯️ regardez egalement ces video pour mieux comprendre le sujet"):
    st.video("https://www.youtube.com/watch?v=GXRU73GeH6E")
    st.video("https://www.youtube.com/watch?v=WIWqwkuV-aM")    
    st.video("https://fast.wistia.net/embed/iframe/b1ljo37bmy")
    


    

  

def display_media():
    st.subheader("Prediction")
    st.markdown("""L'importance du VIX dans le marché boursier est capitale,
                sa prédiction permettra à de nombreux agents financiers de faire leurs choix et de connaître les différentes tendances du marché.
                """)
    image = Image.open("Vix_test_predictions.png")
    st.image(image, width=750)
    st.markdown("""L'importance du VIX dans le marché boursier est cruciale,
                car sa prédiction permet aux agents financiers de prendre des décisions éclairées et de comprendre les tendances du marché.
                Cependant, comme le montre le graphique ci-dessus, il est important de noter que les prévisions ne sont jamais aussi simples qu'il n'y paraît,
                même avec des données de qualité, 
                il est fréquent de ne pas obtenir les résultats escomptés. """)
    matrice2 = Image.open("Matrice_2m(156).png")
    st.image(matrice2, caption="matrice de confusion pour 2 mois", width=500)
    
    matrice15 = Image.open("Matrice_15j(155).png")
    st.image(matrice15, caption="matrice de confusion pour 15 jours", width=500)
    st.markdown(""" Il est clair que, comme l'indiquent ces matrices, le temps a un impact considérable sur l'évolution de nos prédictions. 
                Plus nous essayons de projeter loin dans nos prévisions, plus celles-ci deviennent incertaines.
                Il est donc important de prendre en compte cette incertitude lors de la prise de décision 
                et de ne pas se baser uniquement sur des prévisions à long terme pour éviter les erreurs d'interprétation.""")
    
    images = Image.open("Prediction_trainset(157).png")
    st.image(images, width=750)
    
display_media()


    
def layout():

    
    #sidebar
    st.sidebar.title("La création de cet indice boursier par le CBOE")
    
    st.sidebar.write("""Le VIX a été créé en 1993 par le CBOE (Chicago Board of Options Exchange) 10 ans après l’émergence des premières options sur le S&P 100 et S&P 500.
    À sa naissance il suivait le S&P 100, mais en 2003 la méthode de calcul a été modifiée afin de suivre le S&P 500.
    Désormais de nombreuses bourses autres que le CBOE ont des indices de volatilité.""")
    
    st.sidebar.title("Difficile à utiliser")
    
    st.sidebar.write("""Ceux qui veulent investir sur le VIX devront passer par le marché de produits dérivés,
    qui est difficile à comprendre. Aussi, il faudra mettre en place une stratégie pertinente qui devra certainement
    aller au-delà de vendre la volatilité lorsqu’elle est haute et l’acheter lorsqu’elle est basse.
    Une stratégie que l’on voit relativement souvent.
    Cette stratégie consiste à vendre une assurance lorsqu’elle est chère (en période de krachs) et s’assurer lorsque tout va bien ! 
    La façon dont fonctionnent les contrats à terme met à mal cette stratégie.
    Je vous souhaite le meilleur pour votre épargne et surtout pour tout le reste !""")
    
    

layout()
 

from statsmodels.tsa.arima.model import ARIMA 
def forecast():
    data = st.file_uploader("dataset_VIX.csv", type=["csv"])
    if data is not None:
        data = pd.read_csv("dataset_VIX.csv")
        st.write("Shape of dataset", df.shape)
        data = df["Close"] # Select the column of interest
        st.line_chart(data)
        p = d = q = st.slider("p, d, q", 1, 0, 1  )
        #d =  st.sidebar.slider("d", 0, 5, step = 1)
        #q =  st.sidebar.slider("q", 0, 5, step = 1)
        #day = st.slider("Select number of days",0,5,step = 1)
        model = ARIMA(data, order=(1, 0, 1))
        model_fit = model.fit()
        st.write("Summary of model: ", model_fit.summary())
        n_steps = st.number_input("Enter the number of steps for forecast", 0,15, step = 1)
        
        submitted = st.form_submit_button("Predict")
            
        #if n_steps:
         # , _, _   le remettre derriere le dorecast si besoin    
        forecast= model_fit.forecast(n_steps)
        
        st.line_chart(forecast)
        

st.title("Forecast")
st.form("Forecast")
with st.form("prediction"):
    start_period = st.date_input("Select a start date you want to see your trend")
    end_period = st.date_input("Select an end date you want to see your trend")
    predict = st.form_submit_button("predict")
    forecast()



            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    





#st.write('You selected:', options)

#days = st.slider("Select number of days",1,90,step = 1)
     

    
#st.header("Prediction")
#st.subheader("images")
#image = Image.open("DEMODAY_VIX/Vix_test_prediction(153).png")
#st.markdown("""comment nous pouvons le voir""")

## Run the below code if the check is checked ✅







