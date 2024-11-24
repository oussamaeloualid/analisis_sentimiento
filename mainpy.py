#----------------------------------------------Imports----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from tqdm import tqdm

# Download required NLTK data
'''
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
'''

# Set plot style
plt.style.use('ggplot')

#----------------------------------------------Functions----------------------------------------------

def load_data(filepath, nrows):
    """caraga CSV data y muestra el header dependiendo 
    de la entrada nrows sale un numero de lineas."""
    df = pd.read_csv(filepath)
    df_head = df.head(nrows)
    print(df.head())
    return df_head

def tokenize_and_tag(text):
    """Tokenize and tag parts of speech for the given text."""
    tokens = nltk.word_tokenize(text)
    print("Tokens:", tokens[:10])
    tagged = nltk.pos_tag(tokens)
    print("Tagged:", tagged[:10])
    entities = nltk.chunk.ne_chunk(tagged)
    entities.pprint()
    return tokens, tagged

def plot_review_distribution(df):
    """Plot histogram of review scores."""
    ax = df["Score"].value_counts().sort_index().plot(kind="bar", title="Count of Reviews by Stars", figsize=(10,5))
    ax.set_xlabel('Review Stars')
    plt.show()

def analyze_vader_sentiment(text):
    """Return VADER sentiment scores for a given text."""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def analyze_text_sentiment_vader(df):
    """Analyze sentiment for each text using VADER."""
    sia = SentimentIntensityAnalyzer()
    results = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        results[myid] = sia.polarity_scores(text)
    vader_df = pd.DataFrame(results).T
    vader_df = vader_df.reset_index().rename(columns={'index': 'Id'})
    vader_df = vader_df.merge(df, how='left')
    return vader_df,results

def plot_vader_sentiment(df):
    """Plot VADER sentiment scores by score rating."""


    ax = sns.barplot(data=df, x='Score', y='compound')
    ax.set_title('Compound Score by Amazon Star Review')
    plt.show()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(data=df, x='Score', y='pos', ax=axs[0])
    sns.barplot(data=df, x='Score', y='neu', ax=axs[1])
    sns.barplot(data=df, x='Score', y='neg', ax=axs[2])
    axs[0].set_title('Positive')
    axs[1].set_title('Neutral')
    axs[2].set_title('Negative')
    plt.tight_layout()
    plt.show()

#roberta------------------------------------------------------------
def analyze_roberta_sentiment(text):
    """analisis de sentimentos para un texto  dado usando  RoBERTa.
    esto nos da cuanto es la frecuencia de los diferentes sentimentos neg,pos,pos"""

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
#combinado----------------------------------------------------
def analyze_combined_sentiments(df):
    """combinacion de VADER y RoBERTa sentiment analysis para cada texto en el DataFrame."""
    res = {}
    sia = SentimentIntensityAnalyzer()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
            roberta_result = analyze_roberta_sentiment(text)
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        except RuntimeError:
            print(f'Broke for id {myid}')
    combined_df = pd.DataFrame(res).T
    combined_df = combined_df.reset_index().rename(columns={'index': 'Id'})
    combined_df = combined_df.merge(df, how='left')
    return combined_df




#dibujar combinado-------------------------------------------------
def plot_combined_sentiment(combined_df):
    """Graficar relaciones por pares de los resultados 
       combinados del análisis de sentimientos."""
    sns.pairplot(data=combined_df,
                vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
                hue='Score',
                palette='tab10')
    plt.show()

def run_sentiment_pipeline(text):
    """Ejecutar el análisis de sentimiento con el pipeline de Hugging Face en el texto.
        funcion que vamos a usar para ver el sentimento detras de cada comentario"""
    sent_pipeline = pipeline("sentiment-analysis")
    return sent_pipeline(text)



def filtrar_por_score_negativo(file,numero,strg):
    '''
    esta funcion recibe como parametros de entrada el fichero, el numero de lineas que queremos 
    tratar y el filtro si queremos filtrar por el sentimento NEGATIVE o POSITIVE y nos devuelve los comentarios filtrados
    tambien la lista de las salidas que ha dado la funcion 'run_sentiment_pipeline' 
    por cada iteracion y la cantidad de comentarios filtrados devueltos
    '''
    res=[]
    df=load_data(file,numero)
    i=0
    lista=[]
    for i in range(numero):
        texto=df['Text'].values[i]
        #print(i)
        dicc=run_sentiment_pipeline(texto)
        if(dicc[0]['label']==strg):
            res.append(texto)
            lista.append(dicc[0])
    return res,lista,len(lista)
    
def plot_sentiment(df):
    """
    esta funcion  recibe como entrada el dataframe y devuelve como salida 
    un figura que des¡muestra cuanto por ciento es cada uno de los 
    diferentes sentimentos "pos,neg,neu "
    asi podemos ver si la mayoria de los clientes estan satisfados de los productos o no
    """
    i=0
    lista=[]
    for i in range(len(df)):
        texto=df['Text'].values[i]
        dicc=run_sentiment_pipeline(texto)
        lista.append(dicc[0]['label'])
    sentiment_counts = pd.Series(lista).value_counts()#------------------------
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, figsize=(7, 7))
    plt.title('distrubucion de sentimentos')
    plt.ylabel('')  # Hide y-axis label
    plt.show()


