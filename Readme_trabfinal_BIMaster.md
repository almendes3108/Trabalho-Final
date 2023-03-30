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


![Site de Chaves do Twitter](C:\Users\furnas.DESKTOP-H7LG5SA\OneDrive - furnas.com.br\Documentos\CURSOS\Pos_PUC_BIMASTER\Turmas_BiMaster\Trabalho Final\Readme\token.png)


E através do comando abaixo, conseguimos a autenticação necessária para acessar a API do Twitter :

    #Chave
    bearer_token = "<bearer token gerada no site de desenvolvimento>"

    client = tweepy.Client(bearer_token=bearer_token)

Após a autenticação, usamos o método da API  V2 do Twitter “search_recent_tweets”, para extração de 2000 registros em texto e armazenamos em um dataframe e realizamos em conjunto, a análise de sentimento de textos usando o TextBlob.

    public_tweets = tweepy.Paginator(client.search_recent_tweets, query='eletrobras',  max_results=100).flatten(limit=2000)

Search_recent_tweets: O endpoint de pesquisa recente retorna Tweets dos últimos sete dias que correspondem a uma consulta de pesquisa.
OBS: O código acima permite que tenhamos um limite de 2000 linhas retornadas.


	Com o apoio da biblioteca Pandas, extraímos os textos e importamos em um dataframe contendo inicialmente as colunas ID e texto.

! [Exemplo de Dataframe](C:\Users\furnas.DESKTOP-H7LG5SA\OneDrive - furnas.com.br\Documentos\CURSOS\Pos_PUC_BIMASTER\Turmas_BiMaster\Trabalho Final\Readme\dataframe.png)
