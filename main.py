import requests
from time import sleep
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


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

    # Cuvântul de referință
    # target_word = "warrior"

    # Calculăm embedding-urile
    embeddings1 = model.encode(word_texts)
    embedding2 = model.encode([word])

    # Calculăm similaritățile cosinus
    similarities = cosine_similarity(embeddings1, embedding2).flatten()

    # Sortăm cuvintele după similaritate
    sorted_indices = np.argsort(similarities)[::-1]  # Sortare descrescătoare
    top_10 = [(word_texts[i], similarities[i]) for i in sorted_indices[:10]]

    # Afișare rezultate
    # print(f"Top 10 cuvinte similare cu '{word}':\n")

    return top_10[0][0]

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
