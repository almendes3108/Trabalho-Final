# Análise de Sentimento de Avaliações de Dados Extraídos da Base do Twitter Relacionado ao Tema Eletrobrás

**Aluno: André Luis Mendes Teixeira - Matrícula: 211.101.300**

**Orientadora: Evelyn Batista**

**Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master/) como pré-requisito para conclusão de "Curso de Pós Graduação Business Intelligence Master" na Pontifícia Universidade Católica do Rio de Janeiro**


### **Introdução**

A extração de dados do Twitter pode ser uma ferramenta valiosa para as empresas públicas que buscam compreender melhor o sentimento e as necessidades dos cidadãos. Com bilhões de tweets sendo publicados diariamente, o Twitter oferece uma riqueza de informações que as empresas públicas podem usar para melhorar a visão e o objetivo da empresa para a sociedade.
No caso da Eletrobrás, a extração de dados do Twitter pode ser uma ferramenta de apoio importante permitindo que a empresa compreenda melhor a opinião pública sobre a sua atuação no setor de energia e avalie a efetividade de suas estratégias de comunicação, além de se manter atualizada sobre as tendências do setor de energia.


### **Ferramentas utilizadas**

Para a consulta , extração e tratamento dos dados, usamos as seguintes ferramentas:
- Google Colaboratory: Também conhecido como “Colab”, é um serviço de nuvem gratuito hospedado pelo Google para incentivar a pesquisa de aprendizado de máquina e inteligência artificial. 
- Linguagem Python: é uma linguagem de programação interpretada; ou seja, não precisa ser compilada antes de ser executada, tornando de fácil escrita e teste. A linguagem tem sido muito utilizada em diversas áreas de tecnologia e muito difundida em análise de dados e inteligência artificial.


### **Bibliotecas**

As bibliotecas listadas abaixo foram importadas para possibilitar a extração, tratamento e  análise dos dados, além do treinamento de modelos.

    import tweepy
    import numpy as np
    import pandas as pd
    import re
    import string
    import matplotlib.pyplot as plt
    from textblob import TextBlob
    #from google.colab                    import drive
    from nltk.corpus                     import stopwords
    nltk.download('punkt')
    from nltk.tokenize                   import word_tokenize
    import nltk
    nltk.download('vader_lexicon')
    from collections import Counter
    from nltk.tag import pos_tag
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes             import MultinomialNB
    from sklearn                         import metrics
    from sklearn.model_selection         import cross_val_predict
    from sklearn.ensemble                import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score


### **Extração dos dados**

Para realizar a extração dos dados para este estudo, usamos a biblioteca do Python TWEEPY que serve para acessar a API do Twitter, A versão 2 do Tweepy introduziu um novo método chamado “search_recent_tweets”, que permite pesquisar tweets recentes com base em determinados parâmetros de pesquisa

    !pip3 install tweepy --upgrade

 Para acessar a API do Twitter , primeiramente precisamos criar/ter uma conta no Twitter em seu site de desenvolvimento, [Twitter Developers](https://developer.twitter.com/en/portal/dashboard).


![Site de Chaves do Twitter](https://github.com/almendes3108/Trabalho-Final/blob/pictures/token.png)


E através do comando abaixo, conseguimos a autenticação necessária para acessar a API do Twitter :

    #Chave
    bearer_token = "<bearer token gerada no site de desenvolvimento>"

    client = tweepy.Client(bearer_token=bearer_token)

Após a autenticação, usamos o método da API  V2 do Twitter “search_recent_tweets”, para extração de 2000 registros em texto e armazenamos em um dataframe e realizamos em conjunto, a análise de sentimento de textos usando o TextBlob.

    public_tweets = tweepy.Paginator(client.search_recent_tweets, query='eletrobras',  max_results=100).flatten(limit=2000)

Search_recent_tweets: O endpoint de pesquisa recente retorna Tweets dos últimos sete dias que correspondem a uma consulta de pesquisa.
OBS: O código acima permite que tenhamos um limite de 2000 linhas retornadas.


Com o apoio da biblioteca Pandas, extraímos os textos e importamos em um dataframe contendo inicialmente as colunas ID e texto.

![Exemplo de Dataframe](https://github.com/almendes3108/Trabalho-Final/blob/pictures/dataframe.png)


### **Tratamento dos dados**

Após a extração dos dados, existiu a necessidade de realizar um saneamento dos dados, uma vez que o Twitter é um campo de texto livre e muitas escritas podem vir de diversas formas, por isso, usamos ferramentas como “re.sub” e o “re.escape” que usam expressões regulares para remoção de pontuações e palavras que não tenha algum sentido. A tokenização também se faz importante neste processo de tratamento dos dados adquiridos pela API.


### **Análise de Sentimento**

A análise de sentimento é importante para rotularmos as saídas dos textos extraídos de nossa base de dados. Para tal função , usamos uma biblioteca de processamento de linguagem natural chamada NLTK (Natural Language Toolkit), importando ferramentas para a possibilidade de análise.

O  resultado da análise foi extraído após a criação de uma função denominada “analisar_sentimento2”:

    def analisar_sentimento2(texto):
        tokens = TweetTokenizer(texto)
        score = sia.polarity_scores(texto)
        return score['compound']

![Dataframe_head](https://github.com/almendes3108/Trabalho-Final/blob/pictures/dfhead.png)

Como o resultado é numérico, precisou-se rotular os dados para uma saída categórica:

    # Define uma função de mapeamento
    def mapear_valor(valor):
        if valor > 0:
            return 'Positivo'
        elif valor < 0:
            return 'Negativo'
        else:
            return 'Neutro'

    df2['rotulos'] = df2['Pontuacao2'].apply(mapear_valor)

Onde: valores maiores que zero atribuímos no rótulo o valor “positivo”, valores menores que zero atribuímos no rótulo o valor “negativo” e valores iguais que zero atribuímos no rótulo o valor “neutro”

![Dataframe_rotulos](https://github.com/almendes3108/Trabalho-Final/blob/pictures/df_com_rotulo.png)


### **Nuvem de palavras**

Wordcloud – em português, nuvem de palavras ou nuvem de tags – é um tipo de visualização de dados muito poderoso e ferramenta de Data Science usado quando estamos trabalhando com textos, documentos, pesquisas, entre outras.
Resumidamente, é como se você estivesse contando a frequência com que cada palavra aparece em um texto. Com essa frequência, você define tamanhos proporcionais às palavras, dispondo-as, também, em ângulos diferentes.

![Wordcloud](https://github.com/almendes3108/Trabalho-Final/blob/pictures/workcloud.png)


### **Treinamento de modelos**

O primeiro modelo usado foi o algoritmo Multinomial Naive Bayes que é um algoritmo de aprendizado de máquina utilizado em problemas de classificação de texto ou documentos que envolvem mais de 2 categorias. 

Todos os modelos obtiveram a separação de datasets em treinamento (80%) e em teste (20%).

    X_train, X_test, y_train, y_test = train_test_split(df2['text'], df2['rotulos'], test_size=0.2)

    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = stop_words, max_features = 5000)

    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train_vectors, y_train)

    y_pred_mnb = mnb.predict(X_test_mnb_vectors)


O segundo modelo foi o algoritmo Multnomial usando Bigrams para tentar obter resultados mais precisos nos resultados de classificação.

    vectorizer = CountVectorizer(ngram_range = (1, 2),analyzer = "word", tokenizer = None, preprocessor = None,stop_words = stop_words, max_features = 5000)

    X_train_mnb2_vectors = vectorizer.fit_transform(X_train)
    X_test_mnb2_vectors = vectorizer.transform(X_test)

    modelo = MultinomialNB()
    modelo.fit(X_train_mnb2_vectors, y_train)

    y_pred_mnb2 = modelo.predict(X_test_mnb2_vectors)

O terceiro modelo foi o Random Forest, este algoritmo cria várias árvores de decisão aleatórias a partir de subconjuntos aleatórios do conjunto de dados original, criando, assim, uma floresta de árvores. 


    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = stop_words, max_features = 5000)

    train_data_features = vectorizer.fit_transform(X_train)

    forest = RandomForestClassifier(n_estimators = 100)
    class_sentimentos = y_train.values

    forest = forest.fit(train_data_features, class_sentimentos)

    test_data_features_resultados = vectorizer.transform(X_test)
    resultados = forest.predict(test_data_features_resultados)



O quarto modelo é o Support Vector Model (SVM) é um algoritmo de aprendizado de máquina supervisionado que pode ser usado para desafios de classificação ou regressão.

    
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = stop_words, max_features = 5000)

    X_train_svm_vectors = vectorizer.fit_transform(X_train)
    X_test_svm_vectors = vectorizer.transform(X_test)
    svm = SVC(kernel='linear')
    svm.fit(X_train_svm_vectors, y_train)



### **Resultados**

Abaixo, são apresentados as matrizes e os valores de acurácia de cada um dos modelos apresentados: 

1)	Multinomial Naive Bayes

![multinomial](https://github.com/almendes3108/Trabalho-Final/blob/pictures/matriz_multinomial.png)

Acurácia: 0,92

2)	Multinomial usando Bigrams

![multinomialbigrams](https://github.com/almendes3108/Trabalho-Final/blob/pictures/matriz_multinomial_bigram.png)

Acurácia: 0,9075

3)	Random Forest

![randomforest](https://github.com/almendes3108/Trabalho-Final/blob/pictures/matriz_Randomforest.png)

Acurácia: 0,9825

4)	SVM

![svm](https://github.com/almendes3108/Trabalho-Final/blob/pictures/matriz_svm.png)

Acurácia: 0,9875



### **Conclusão**

Após o treinamento dos 4 modelos de classificação de textos, o algoritmo Support Vector Machine (SVM) foi o que se mostrou mais eficaz tendo uma acurácia superior aos demais algoritmos apresentados nesse estudo.  

Esse trabalho foi concebido através de muitos conhecimentos adquiridos do Curso de Pós-Graduação Business Intelligence Master usando conceitos de Python adquiridos nos módulos de Sistemas de Apoio à Decisão, conceitos de Localização e Uso de Informação e muitas atividades envolvidas em módulos de Processamento de Linguagem Natural e DataMining.

