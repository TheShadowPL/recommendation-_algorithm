# main.py

from semantic_map import (
    preprocess_tags, augment_data,
    encode_tags_bert, print_similar_tags,
    save_encoded_tags, load_encoded_tags, encode_all_tags
)
from recommendation_algorithm import ocena_podobienstwa_artykułu, ocena_koncowa_artykułu
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
import json
from datetime import datetime
import pickle

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data['wp_articles_meta'])
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m-%d-%Y')
    return df

def load_user_profile(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data['user_profiles'])

def get_user_article_tags(user_profiles_df, articles_df, user_identifier):
    user_articles = user_profiles_df[user_profiles_df['identifier'] == user_identifier]['article_ID'].tolist()
    user_tags = articles_df[articles_df['article_id'].isin(user_articles)]['meta_tags'].tolist()
    user_tags = [tag for sublist in user_tags for tag in sublist]
    return user_tags

def recommend_articles_for_user(user_identifier, user_profiles_df, articles_df, encoded_tags, X_stopnia_podobienstwa, X, Y, Z):
    user_tags = get_user_article_tags(user_profiles_df, articles_df, user_identifier)
    recommendations = []

    for _, article in articles_df.iterrows():
        article_tags = article['meta_tags']
        ocena_podobienstwa = ocena_podobienstwa_artykułu(user_tags, article_tags, encoded_tags, X_stopnia_podobienstwa)
        ocena_koncowa = ocena_koncowa_artykułu(ocena_podobienstwa, article['publication_date'], article['average_rating'], len(user_tags), X, Y, Z)
        recommendations.append((article['article_id'], ocena_koncowa))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:5]  # tu macie parametr do zwracania ilosci rekomendowanych artykulow

if __name__ == "__main__":
    articles_file_path = 'db.json'
    user_profiles_file_path = 'user_profiles.json'
    encoded_tags_path = 'encoded_tags.pkl'

    articles_df = load_data(articles_file_path)
    user_profiles_df = load_user_profile(user_profiles_file_path)

    articles_df = preprocess_tags(articles_df)
    articles_df = augment_data(articles_df)

    if os.path.exists(encoded_tags_path):
        encoded_tags = load_encoded_tags(encoded_tags_path)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        encoded_tags = encode_all_tags(articles_df, tokenizer, model)
        save_encoded_tags(encoded_tags_path, encoded_tags)

    user_identifiers = ['user1']
    X_stopnia_podobienstwa = 0.7
    X = 1.0
    Y = 5
    Z = 1.0

    for user_identifier in user_identifiers:
        recommendations = recommend_articles_for_user(user_identifier, user_profiles_df, articles_df, encoded_tags, X_stopnia_podobienstwa, X, Y, Z)
        print(f"Rekomendacje dla {user_identifier}:")
        for article_id, score in recommendations:
            print(f"article_id: {article_id}, Ocena: {score:.2f}")
        print("\n")
