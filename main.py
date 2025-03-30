import requests
from time import sleep
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
from ollama import ChatResponse
import numpy as np
import re



host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5


def what_beats(word):
    # Inițializare model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Lista de cuvinte
    word_texts = [
        "Feather", "Coal", "Pebble", "Leaf", "Paper", "Rock", "Water", "Twig",
        "Sword", "Shield", "Gun", "Flame", "Rope", "Disease", "Cure", "Bacteria",
        "Shadow", "Light", "Virus", "Sound", "Time", "Fate", "Earthquake", "Storm",
        "Vaccine", "Logic", "Gravity", "Robots", "Stone", "Echo", "Thunder", "Karma",
        "Wind", "Ice", "Sandstorm", "Laser", "Magma", "Peace", "Explosion", "War",
        "Enlightenment", "Nuclear Bomb", "Volcano", "Whale", "Earth", "Moon", "Star",
        "Tsunami", "Supernova", "Antimatter", "Plague", "Rebirth", "Tectonic Shift",
        "Gamma-Ray Burst", "Human Spirit", "Apocalyptic Meteor", "Earth’s Core",
        "Neutron Star", "Supermassive Black Hole", "Entropy"
    ]
    
    target_word = word

    embeddings1 = model.encode(word_texts)
    embedding2 = model.encode([target_word])

    similarities = cosine_similarity(embeddings1, embedding2).flatten()

    sorted_indices = np.argsort(similarities)[::-1]  # Sortare descrescătoare
    top_10 = [word_texts[i] for i in sorted_indices[:10]]
    
    response: ChatResponse = chat(model='llama3.2:1b', messages=[
      {
        'role': 'user',
        'content': f"""I will provide you with a list of 10 words. 
                    I will also give you a single word. Your goal is to choose from the provided list 5 words 
                    which logically counter the given single word. Each word should be associated with a confidence 
                    score between 0 and 1 that you calculate based on how sure you are that the word you generate logically counters the word I give you. 
    
                    This is the single word: {target_word}. 
                    This is the list of dictionaries: {top_10}. 
                    Your response should be formatted like this: 
                    [{{word1": "confidence_score1"}}, {{"word2": "confidence_score2"}}, {{"word3": "confidence_score3"}}, {{"word4": "confidence_score4"}}, {{"word5": "confidence_score5"}}]
                 """,
      }
    ])
    
    pattern = r'{"(\w+)": ([0-9.]+)}'
    matches = re.findall(pattern, response['message']['content'])
    extracted_data = [{word: float(score)} for word, score in matches]

    if extracted_data == [] :
        return top_10[0]
    filtered = [word for item in extracted_data for word, score in item.items() if score > 0.5]
    if filtered == [] :
        if extracted_data == []:
            return top_10[0]
        else:
            max_word = list(max(data, key=lambda x: list(x.values())[0]).keys())[0]
            return max_word

    top_10_indices = {word: i for i, word in enumerate(top_10)}
    sorted_filtered = sorted(filtered, key=lambda x: top_10_indices.get(x, float('inf')))
    return sorted_filtered[0]
    
def play_game(player_id):
    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            print(status.json())
        
        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(response.json())

play_game("eQBWh7zU1E")
