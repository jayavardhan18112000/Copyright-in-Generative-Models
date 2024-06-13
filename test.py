# import streamlit as st
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch
# import nltk
# from nltk.util import ngrams
# from nltk.lm.preprocessing import pad_sequence
# from nltk.probability import FreqDist
# import plotly.express as px
# from collections import Counter
# from nltk.corpus import stopwords
# import string

# nltk.download('punckt')
# nltk.download('stopwords')
# # Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# def calculate_perplexity(text):
#     encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
#     input_ids = encoded_input[0]

#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits

#     perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
#     return perplexity.item()

# def calculate_burstiness(text):
#     tokens = nltk.word_tokenize(text.lower())
#     word_freq = FreqDist(tokens)
#     repeated_count = sum(count > 1 for count in word_freq.values())
#     burstiness_score = repeated_count / len(word_freq)
#     return burstiness_score

# def calculate_lexical_diversity(text):
#     tokens = nltk.word_tokenize(text.lower())
#     return len(set(tokens)) / len(tokens) if tokens else 0

# def calculate_sentence_complexity(text):
#     sentences = nltk.sent_tokenize(text)
#     average_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
#     return average_length

# def plot_top_repeated_words(text):
#     # Tokenize the text and remove stopwords and special characters
#     tokens = text.split()
#     stop_words = set(stopwords.words('english'))
#     tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

#     # Count the occurrence of each word
#     word_counts = Counter(tokens)

#     # Get the top 10 most repeated words
#     top_words = word_counts.most_common(10)

#     # Extract the words and their counts for plotting
#     words = [word for word, count in top_words]
#     counts = [count for word, count in top_words]

#     # Plot the bar chart using Plotly
#     fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Repeated words for Top 10')
#     st.plotly_chart(fig, use_container_width=True)


# st.set_page_config(layout="wide")

# st.title("Alerting System and Human Review")
# text_area = st.text_area("Type your content here", "")

# if text_area is not None:
#     if st.button("Check"):
#         col1, col2, col3 = st.columns([1,1,1])
#         with col1:
#             st.info("Your Text")
#             st.success(text_area)
        
#         with col2:
#             st.info("Detection Score")
#             perplexity = calculate_perplexity(text_area)
#             burstiness_score = calculate_burstiness(text_area)
#             lexical_diversity = calculate_lexical_diversity(text_area)
#             sentence_complexity = calculate_sentence_complexity(text_area)
            

#             st.write("Perplexity:", perplexity)
#             st.write("Burstiness Score:", burstiness_score)
#             st.write("lexical diversity:", lexical_diversity)
#             st.write("sentence complexity:", sentence_complexity)

#             if perplexity > 30000 and burstiness_score < 0.25:
#                 st.error("Alert and Warning")
#             else:
#                 st.success("No need to worry")
            
            
            
            
#         with col3:
#             st.info("The Metrics Analysis")
#             plot_top_repeated_words(text_area)













import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def preprocess_text(text):
    text = re.sub(r'\".+?\"', '', text)
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def calculate_similarity(essay1, essay2):
    preprocessed_essay1 = preprocess_text(essay1)
    preprocessed_essay2 = preprocess_text(essay2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_essay1, preprocessed_essay2])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity
def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def calculate_lexical_diversity(text):
    tokens = nltk.word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens) if tokens else 0

def calculate_sentence_complexity(text):
    sentences = nltk.sent_tokenize(text)
    average_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
    return average_length
st.set_page_config(layout="wide")
st.title("Human Alerting and Review System")
st.markdown("""
#### Threshold Values for Metrics (All conditions must be met for AI identification)
- **Perplexity:** Greater than 30000
- **Burstiness:** Less than 0.25
- **Lexical Diversity:** Less than 0.71
- **Sentence Complexity:** Between 15 and 25
""")

col1, col2 = st.columns(2)
with col1:
    text_area1 = st.text_area("Type your content here", "", key="1")
with col2:
    text_area2 = st.text_area("Type your content here", "", key="2")

if st.button("Analyze Texts"):
    metrics1 = {
        "Perplexity": calculate_perplexity(text_area1),
        "Burstiness": calculate_burstiness(text_area1),
        "Lexical Diversity": calculate_lexical_diversity(text_area1),
        "Sentence Complexity": calculate_sentence_complexity(text_area1)
    }

    metrics2 = {
        "Perplexity": calculate_perplexity(text_area2),
        "Burstiness": calculate_burstiness(text_area2),
        "Lexical Diversity": calculate_lexical_diversity(text_area2),
        "Sentence Complexity": calculate_sentence_complexity(text_area2)
    }
    def determine_origin(metrics):
        if (metrics["Perplexity"] > 30000 and metrics["Burstiness"] < 0.25 and
            metrics["Lexical Diversity"] < 0.71 and 15 <= metrics["Sentence Complexity"] < 25):
            return "AI Generated"
        else:
            return "Human Generated"

    origin1 = determine_origin(metrics1)
    origin2 = determine_origin(metrics2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Text 1 Metrics")
        for key, value in metrics1.items():
            st.metric(label=key, value=f"{value:.2f}")
        st.write(f"Origin: {origin1}")
    with col2:
        st.subheader("Text 2 Metrics")
        for key, value in metrics2.items():
            st.metric(label=key, value=f"{value:.2f}")
        st.write(f"Origin: {origin2}")

    similarity_score = calculate_similarity(text_area1, text_area2)
    st.subheader("Similarity Score between Texts")
    st.metric(label="Similarity Score", value=f"{similarity_score:.2f}")













            





# import os
# import pandas as pd
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch
# import nltk
# from nltk.probability import FreqDist
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re 

# # Setup
# nltk.download('punkt')
# nltk.download('stopwords')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# def preprocess_text(text):
#     text = re.sub(r'\".+?\"', '', text)

#     stop_words = set(stopwords.words("english"))
#     words = word_tokenize(text.lower())
#     filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
#     return " ".join(filtered_words)

# def calculate_similarity(essay1, essay2):
#     preprocessed_essay1 = preprocess_text(essay1)
#     preprocessed_essay2 = preprocess_text(essay2)

#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([preprocessed_essay1, preprocessed_essay2])

#     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#     return similarity

# def calculate_perplexity(text):
#     encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
#     input_ids = encoded_input[0]

#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits

#     perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
#     return perplexity.item()

# def calculate_burstiness(text):
#     tokens = nltk.word_tokenize(text.lower())
#     word_freq = FreqDist(tokens)
#     repeated_count = sum(count > 1 for count in word_freq.values())
#     burstiness_score = repeated_count / len(word_freq)
#     return burstiness_score

# def calculate_lexical_diversity(text):
#     tokens = nltk.word_tokenize(text.lower())
#     return len(set(tokens)) / len(tokens) if tokens else 0

# def calculate_sentence_complexity(text):
#     sentences = nltk.sent_tokenize(text)
#     average_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
#     return average_length

# def process_texts(original_dir, rewritten_dir):
#     original_files = os.listdir(original_dir)
#     rewritten_files = os.listdir(rewritten_dir)

#     data = []
#     for original_file, rewritten_file in zip(sorted(original_files), sorted(rewritten_files)):
#         original_path = os.path.join(original_dir, original_file)
#         rewritten_path = os.path.join(rewritten_dir, rewritten_file)

#         with open(original_path, 'r', encoding='utf-8') as file:
#             original_text = file.read()
        
#         with open(rewritten_path, 'r', encoding='utf-8') as file:
#             rewritten_text = file.read()

#         # Calculate metrics
#         perplexity_original = calculate_perplexity(original_text)
#         burstiness_original = calculate_burstiness(original_text)
#         lexical_diversity_original = calculate_lexical_diversity(original_text)
#         sentence_complexity_original = calculate_sentence_complexity(original_text)

#         perplexity_rewritten = calculate_perplexity(rewritten_text)
#         burstiness_rewritten = calculate_burstiness(rewritten_text)
#         lexical_diversity_rewritten = calculate_lexical_diversity(rewritten_text)
#         sentence_complexity_rewritten = calculate_sentence_complexity(rewritten_text)

#         similarity_score = calculate_similarity(original_text, rewritten_text)

#         data.append({
#             'Type': 'Original',
#             'File': original_file,
#             'Perplexity': perplexity_original,
#             'Burstiness': burstiness_original,
#             'Lexical Diversity': lexical_diversity_original,
#             'Sentence Complexity': sentence_complexity_original,
#             'Similarity Score': similarity_score
#         })
        
#         data.append({
#             'Type': 'AI Generated',
#             'File': rewritten_file,
#             'Perplexity': perplexity_rewritten,
#             'Burstiness': burstiness_rewritten,
#             'Lexical Diversity': lexical_diversity_rewritten,
#             'Sentence Complexity': sentence_complexity_rewritten,
#             'Similarity Score': similarity_score
#         })

#     return data

# # Paths to the directories containing the original and rewritten stories
# original_dir = 'C:/Jayavardhan/5602/Project/Original_Stories'
# rewritten_dir = 'C:/Jayavardhan/5602/Project/Rewritten_Stories'

# all_data = process_texts(original_dir, rewritten_dir)
# df = pd.DataFrame(all_data)

# # Export to Excel
# df.to_excel('text_analysis4.xlsx', index=False)









            
