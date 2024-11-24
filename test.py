from mainpy import *


df=load_data('Reviews.csv',10)



def test_tokenize_and_tag():
    ejemplo = df['Text'][50]
    tokenize_and_tag(ejemplo)

def test_plot_sentiment():
    plot_sentiment(df)


def test_analyze_vader_sentiment():
    print(analyze_vader_sentiment("I am so happy!"))
    print(analyze_vader_sentiment("This is the worst thing ever."))
    print(analyze_vader_sentiment(df["Text"].values[0]))

def test_run_sentiment_pipeline():
    print(run_sentiment_pipeline("j'aime bien sentiment analysis"))
#---------------------------plot---------------------
def test_plot_review_distribution():
    plot_review_distribution(df)


def test_plot_vader_sentiment():
    vader_df = analyze_text_sentiment_vader(df)
    plot_vader_sentiment(vader_df[0])

def test_plot_combined_sentiment():
    combined_df = analyze_combined_sentiments(df)
    plot_combined_sentiment(combined_df)


#------------------------filter-test------------------------
def test_filtrar_por_score_negativo():
    print(filtrar_por_score_negativo('data/Reviews.csv',4,"NEGATIVE")[0]
    ,"\n",filtrar_por_score_negativo('data/Reviews.csv',4,"NEGATIVE")[1],"\n",filtrar_por_score_negativo('Reviews.csv',4,"NEGATIVE")[2])
    



if __name__ == "__main__":
    pass
    #descomentar para probar cada una de las funciones 
    #test_filtrar_por_score_negativo()
    #test_plot_review_distribution()
    #test_analyze_vader_sentiment()
    #test_tokenize_and_tag()
    #test_plot_sentiment()
    #test_plot_vader_sentiment()
    #test_plot_combined_sentiment()
    #test_run_sentiment_pipeline()
    