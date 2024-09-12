from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Carica il file CSV
data = pd.read_csv('data.csv')
domande = data['Domande']

# Trasforma le domande in TF-IDF embeddings
vectorizer = TfidfVectorizer()
embed = vectorizer.fit_transform(domande)

@app.route("/", methods=["GET", "POST"])
def index():
    result = []
    if request.method == "POST":
        question = request.form["question"]
        question_embedding = vectorizer.transform([question])
        similarity = cosine_similarity(question_embedding, embed)
        
        # Trova le 3 domande con la massima similarità
        lista_indici_domande = []
        for i in range(3):
            max_index = np.argmax(similarity[0])
            lista_indici_domande.append(domande[max_index])
            similarity[0, max_index] = -np.inf  # Esclude la domanda selezionata

        result = lista_indici_domande

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
