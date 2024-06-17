# semantic_map.py

import pickle
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertModel
import torch
import difflib

def preprocess_tags(df):
    df['meta_tags'] = df['meta_tags'].apply(lambda x: x if isinstance(x, list) else [x])
    return df

def augment_data(df):
    augmented_tags = []
    for tags in df['meta_tags']:
        augmented_tags_article = []
        augmented_tags_article.extend(tags)
        for tag in tags:
            synsets = wordnet.synsets(tag)
            for syn in synsets:
                synonyms = syn.lemma_names()
                augmented_tags_article.extend(synonyms)
        augmented_tags.append(augmented_tags_article)
    df['augmented_tags'] = augmented_tags
    return df

def encode_tags_bert(tokenizer, model, tags):
    flat_tags = [item for sublist in tags for item in sublist]

    max_len = max(len(tag) for tag in flat_tags)
    encoded_tags = []
    for tag in flat_tags:
        tokenized_input = tokenizer(tag, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        with torch.no_grad():
            outputs = model(**tokenized_input)
        encoded_tag = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        encoded_tags.append(encoded_tag)
    max_len_tag = max(len(tag) for tag in encoded_tags)
    for i, tag in enumerate(encoded_tags):
        if len(tag) < max_len_tag:
            padding_length = max_len_tag - len(tag)
            padded_tag = np.pad(tag, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
            encoded_tags[i] = padded_tag
    return np.array(encoded_tags)

def save_encoded_tags(file_path, encoded_tags):
    with open(file_path, 'wb') as file:
        pickle.dump(encoded_tags, file)
    print(f"Tagi zostały zapisane do pliku {file_path}")

def load_encoded_tags(file_path):
    with open(file_path, 'rb') as file:
        encoded_tags = pickle.load(file)
    print(f"Tagi zostały wczytane z pliku {file_path}")
    return encoded_tags

def print_similar_tags(tag, encoded_tags, augmented_tags, top_n=5):
    flat_tag_list = [item for sublist in augmented_tags for item in sublist]

    try:
        tag_index = flat_tag_list.index(tag)
    except ValueError:
        closest_tag = difflib.get_close_matches(tag, flat_tag_list, n=1)
        if closest_tag:
            tag = closest_tag[0]
            tag_index = flat_tag_list.index(tag)
        else:
            print(f"Tag '{tag}' nie istnieje i nie ma podobnych tagów.")
            return

    tag_vector = encoded_tags[tag_index]

    similarities = np.dot(encoded_tags, tag_vector)
    most_similar_indices = similarities.argsort()[-top_n - 1:-1][::-1]
    least_similar_indices = similarities.argsort()[:top_n]

    print(f"Najbardziej podobne tagi do '{tag}':")
    for i, idx in enumerate(most_similar_indices, 1):
        similarity_score = similarities[idx]
        print(f"{i}. {flat_tag_list[idx]} (odległość: {similarity_score:.4f})")

    print(f"Najmniej podobne tagi do '{tag}':")
    for i, idx in enumerate(least_similar_indices, 1):
        similarity_score = similarities[idx]
        print(f"{i}. {flat_tag_list[idx]} (odległość: {similarity_score:.4f})")

    return most_similar_indices, tag_index

def encode_all_tags(articles_df, tokenizer, model):
    tags = set(tag for tags in articles_df['augmented_tags'] for tag in tags)
    tag_to_vector = {}
    for tag in tags:
        tokenized_input = tokenizer(tag, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokenized_input)
        tag_to_vector[tag] = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    return tag_to_vector
