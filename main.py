from gensim import corpora, models, similarities
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from random import choice
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

intents = {
    'saudacao': {
        'patterns': ['oi', 'olá', 'boa tarde', 'boa noite', 'e aí', 'bom dia', 'ei'],
        'responses': ['Olá, como posso ajudar?', 'Oi, o que posso fazer por você hoje?', 'Olá, em que posso ser útil?']
    },
    'relatar_crime': {
        'patterns': ['quero relatar um crime', 'fui roubado', 'tenho uma denúncia', 'fui assaltado', 'meu carro foi roubado', 'minha casa foi invadida'],
        'responses': ['Por favor, forneça mais detalhes sobre o crime.', 'Qual é a natureza do crime que você gostaria de relatar?', 'Entendi. Por favor, descreva o ocorrido.'],
        'action': 'prompt_details'
    },
    'emergencia': {
        'patterns': ['preciso de ajuda', 'estou em perigo', 'socorro', 'estou sendo atacado', 'alguém está me seguindo', 'ouvi tiros'],
        'responses': ['Estamos enviando ajuda imediatamente. Por favor, fique seguro.', 'Permaneça onde está. Ajuda está a caminho.', 'Entendi. Mantenha-se em um local seguro enquanto enviamos ajuda.'],
        'action': 'prompt_call',
        'options': ['Sim', 'Não']
    },
    'informacoes': {
        'patterns': ['quero informações', 'preciso de informação', 'quero saber sobre...', 'pode me dizer...'],
        'responses': ['Claro, que tipo de informação você está procurando?', 'Posso ajudar. O que você gostaria de saber?', 'Por favor, especifique sua pergunta.']
    },
    'agradecimentos': {
        'patterns': ['obrigado', 'agradeço', 'valeu', 'muito obrigado'],
        'responses': ['De nada! Estou aqui para ajudar.', 'Sempre à disposição.', 'Por nada, fique seguro!']
    },
    'despedida': {
        'patterns': ['tchau', 'até logo', 'adeus', 'até mais'],
        'responses': ['Até mais! Fique seguro.', 'Tchau! Se precisar, estou aqui.', 'Adeus!']
    },
    'orientacao': {
        'patterns': ['o que devo fazer?', 'me ajude', 'não sei o que fazer', 'estou perdido'],
        'responses': ['Vamos resolver isso juntos. Me dê mais detalhes.', 'Fique calmo. Me conte mais sobre a situação.', 'Estou aqui para ajudar. Por favor, explique a situação.']
    },
    'comentario_positivo': {
        'patterns': ['você é ótimo', 'bom trabalho', 'você me ajudou'],
        'responses': ['Fico feliz em poder ajudar!', 'Obrigado pelo feedback!', 'Estou aqui para ajudar.']
    },
    'comentario_negativo': {
        'patterns': ['isso não ajuda', 'você não entendeu', 'não foi isso que perguntei'],
        'responses': ['Peço desculpas. Por favor, reformule ou seja mais específico.', 'Desculpe pelo mal-entendido. Vamos tentar novamente.', 'Vamos resolver isso. Me dê mais detalhes.']
    },
    'especificacao': {
        'patterns': ['estou falando de...', 'me refiro a...', 'quero dizer...'],
        'responses': ['Entendi. Por favor, continue.', 'Obrigado por especificar. Vamos continuar.', 'Ok, agora entendi. Prossiga.']
    }
    #... Adicione mais intenções conforme necessário
}

def get_intent(user_message):
    tokens = word_tokenize(user_message.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    
    scores = {}
    for intent, values in intents.items():
        for pattern in values['patterns']:
            if pattern in filtered_tokens:
                if intent not in scores:
                    scores[intent] = 0
                scores[intent] += 1

    return max(scores, key=scores.get, default=None)

# Processamento para calcular similaridade
documents = [pattern for intent in intents.values() for pattern in intent['patterns']]
texts = [word_tokenize(document.lower()) for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity(lsi[corpus])

def get_intent(user_message):
    vec_bow = dictionary.doc2bow(word_tokenize(user_message.lower()))
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]
    max_similarity = max(sims)
    
    if max_similarity > 0.5:  # Threshold de similaridade
        detected_intent = documents[sims.argmax()]
        for intent, values in intents.items():
            if detected_intent in values['patterns']:
                return intent
    return None

#Antigo
# @app.route('/chatbot', methods=['POST'])
# def chatbot_response():
#     user_message = request.json['message']
#     detected_intent = get_intent(user_message)
    
#     if detected_intent:
#         return jsonify({"response": choice(intents[detected_intent]['responses'])})
#     else:
#         return jsonify({"response": "Desculpe, não entendi. Por favor, seja mais específico."})


@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_message = request.json['message']
    detected_intent = get_intent(user_message)
    
    if detected_intent:
        response = {
            "response": choice(intents[detected_intent]['responses']),
            "intent": detected_intent
        }
        return jsonify(response)
    else:
        return jsonify({"response": "Desculpe, não entendi. Por favor, seja mais específico.", "intent": "none"})


@app.route('/feedback', methods=['POST'])
def receive_feedback():
    feedback = request.json['feedback']
    # Aqui, você pode armazenar o feedback em um banco de dados ou em um arquivo para análise futura.
    return jsonify({"response": "Obrigado pelo feedback!"})

@app.route('/welcome_message', methods=['GET'])
def welcome_message():
    welcome_responses = ["Olá! Como posso ajudar?", "Oi! Em que posso ser útil hoje?", "Ei! Como posso ajudar?"]
    return jsonify({"response": choice(welcome_responses)})

if __name__ == '__main__':
    app.run(debug=True)
