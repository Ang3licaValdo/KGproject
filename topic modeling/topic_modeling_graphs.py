import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from stableversionroberta import abstracts_list

def evaluate_num_topics(abstracts_list, min_topics, max_topics, step):
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(abstracts_list)
    perplexity_scores = []

    for num_topics in range(min_topics, max_topics + 1, step):
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)
        perplexity_scores.append(lda.perplexity(X))
        print(f"Number of Topics: {num_topics}, Perplexity score: {lda.perplexity(X)}")

    # Plot perplexity scores
    plt.plot(range(min_topics, max_topics + 1, step), perplexity_scores, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity Score")
    plt.title("Perplexity Score for Different Numbers of Topics")
    plt.show()
    return perplexity_scores

# Example usage
perplexity_scores = evaluate_num_topics(abstracts_list, min_topics=2, max_topics=30, step=1)


import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def evaluate_num_topics(abstracts_list, min_topics, max_topics, step):
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(abstracts_list)
    perplexity_scores = []

    for num_topics in range(min_topics, max_topics + 1, step):
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)
        perplexity_scores.append(lda.perplexity(X))
        print(f"Number of Topics: {num_topics}, Perplexity score: {lda.perplexity(X)}")

    # Plot perplexity scores
    plt.plot(range(min_topics, max_topics + 1, step), perplexity_scores, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity Score")
    plt.title("Perplexity Score for Different Numbers of Topics")
    plt.show()
    return perplexity_scores

# Example usage
perplexity_scores = evaluate_num_topics(abstracts_list, min_topics=2, max_topics=30, step=1)

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')  # Uncomment this line to download the 'punkt' resource

def compute_coherence(abstracts_list):
    # Tokenize the abstracts
    tokenized_abstracts = [word_tokenize(abstract) for abstract in abstracts_list]
    
    # Create a Gensim Dictionary object
    dictionary = Dictionary(tokenized_abstracts)
    
    # Create a list of term lists for each document
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_abstracts]
    
    coherence_scores = []
    num_topics_range = range(2, 31)  # Range of number of topics to evaluate
    
    for num_topics in num_topics_range:
        lda = LdaModel(corpus, num_topics=num_topics, random_state=0)
        
        # Get the term distribution for each topic
        topic_tokens = [[dictionary[term_id] for term_id, _ in lda.get_topic_terms(topic_id)] for topic_id in range(num_topics)]
        
        # Create a CoherenceModel object with the corpus, dictionary, and topic tokens
        coherence_model = CoherenceModel(
            topics=topic_tokens,
            texts=tokenized_abstracts,
            dictionary=dictionary,
            coherence='c_v'  # You can choose the type of coherence metric ('c_v', 'u_mass', 'c_uci', 'c_npmi')
        )
        
        coherence_scores.append(coherence_model.get_coherence())


    # Return the coherence metric results for each number of topics
    return coherence_scores


# Obtener los puntajes de coherencia
coherence_scores = compute_coherence(abstracts_list)
for i, score in enumerate(coherence_scores):
    num_topics = i + 2  # Adjust the index to start at 2
    print(f"Number of Topics: {num_topics}, Coherence score: {score}")

# Rango de números de temas evaluados
num_topics_range = range(2, 31)

# Crear una figura más grande
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

# Graficar los puntajes de coherencia
plt.plot(num_topics_range, coherence_scores, marker='o', color='orange')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence score by number of topics (higher is better)')

# Agregar retícula
plt.grid(True)

# Establecer los números de temas en el eje x
plt.xticks(num_topics_range, num_topics_range)

plt.savefig("Coherence_plot.png", dpi=300)

# Mostrar el gráfico
plt.show()


import matplotlib.pyplot as plt

min_topics = 2
max_topics = 30
# Crear una figura más grande
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

# Escalando los valores
scaled_perplexity_scores = [(score - min(perplexity_scores)) / (max(perplexity_scores) - min(perplexity_scores)) for score in perplexity_scores]
scaled_coherence_scores = [(score - min(coherence_scores)) / (max(coherence_scores) - min(coherence_scores)) for score in coherence_scores]

# Plotear los valores escalados
plt.plot(scaled_perplexity_scores, label='Perplexity (lower is better)')
plt.plot(scaled_coherence_scores, label='Coherence (higher is better)')

# Marcar los puntos
plt.scatter(range(len(perplexity_scores)), scaled_perplexity_scores, color='blue')
plt.scatter(range(len(coherence_scores)), scaled_coherence_scores, color='orange')

# Configuraciones del gráfico
plt.xlabel('Number of Topics [2-30]')
plt.ylabel('Score')
plt.title('Perplexity and Coherence Scores')
plt.legend()

# Agregar retícula
plt.grid(True)

# Establecer los números de temas en el eje x
plt.xticks(range(len(perplexity_scores)), range(min_topics, max_topics + 1))

plt.savefig("Coherence_vs_Perplexity_plot.png", dpi=300)

# Mostrar el gráfico
plt.show()
