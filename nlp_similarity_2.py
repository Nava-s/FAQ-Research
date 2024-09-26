import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data_riformulate.csv', delimiter = ";")
domande = data['Domande']
riformulazioni = data['Riformulazione']

cs = np.hstack([domande,riformulazioni])
cs = pd.DataFrame(cs)
documents = cs.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()

vectorizer = TfidfVectorizer()
embed = vectorizer.fit_transform(documents)

question = input()
question = vectorizer.transform([question])

Domanda = vectorizer.transform(domande)
Riformulazione = vectorizer.transform(riformulazioni)

similarity = []
similarity_1 = cosine_similarity(question, Domanda)
similarity_2 = cosine_similarity(question, Riformulazione)

for row in range(len(domande)):
    similarity.append(max(similarity_1[0][row],similarity_2[0][row]))

similarity = np.array(similarity).reshape(1,-1)

def similarity_model(similarity_data, domande):
    lista_domande = []
    for i in range(3):
        max_index = np.argmax(similarity_data)
        lista_domande.append(domande[max_index])
        similarity_data[0,max_index] = -1
    return lista_domande
    
faq = similarity_model(similarity, domande)