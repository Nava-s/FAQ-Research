from nlp_similarity_2 import *

data_best = pd.read_csv('data_best.csv',delimiter=';')

# Embedding delle domande sintetiche e delle domande presenti nel dataset
syn_question = vectorizer.transform(data_best['DomandaSintetica'])

# Lista per memorizzare le 3 domande migliori per ogni domanda sintetica
best_questions_list = []   

# Ciclo per calcolare la similarità per ogni domanda sintetica
for i in syn_question:
    # Calcola la similarità tra la domanda sintetica e il set di domande
    similarity_values1 = cosine_similarity(i, embed)
    similarity_values2 = cosine_similarity(i, Riformulazione)
    similarity_values = []
    for j in range(len(domande)):
        max_similarity = max(similarity_values1[0][j],similarity_values2[0][j])
        similarity_values.append(max_similarity)
        # Usa la funzione similarity_model per ottenere le 3 migliori domande
    
    top_3_questions = similarity_model(np.array(similarity_values).reshape(1,-1), domande)
    
    # Aggiungi le 3 migliori domande alla lista
    best_questions_list.append(top_3_questions)

# Confronto con le domande "best" del dataset 'data_best'
correct_matches = 0  # Variabile per contare i match corretti

for i in range(len(best_questions_list)):
    if data_best['DomandaCorretta'][i] in best_questions_list[i]:
        correct_matches += 1


# Output del risultato
print(f"Top3 Accuracy Score is: {round(correct_matches/len(data_best),3)}")

log_error = {
    "DomandaSintetica": [],
    "DomandaCorretta": [],
    "ListaTop3": []
}

for i in range(len(best_questions_list)):
    if data_best['DomandaCorretta'][i] not in best_questions_list[i]:
        log_error["DomandaSintetica"].append(data_best['DomandaSintetica'][i])
        log_error["DomandaCorretta"].append(data_best['DomandaCorretta'][i])
        log_error["ListaTop3"].append(best_questions_list[i])

df_log = pd.DataFrame(log_error)
df_log.to_csv(r'c:\Users\antonio.proietti\OneDrive - A2A Group\Desktop\backup lavoro\File Antonio\ML Projects\CosineSimilarityNLP\log_error.csv', encoding = 'utf-8', index = False)