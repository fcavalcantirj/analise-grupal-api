import emoji
import seaborn as sns
import numpy as np
import re
import os
import base64
import requests
import logging
import cloudinary
import openai
import json
import time
from collections import defaultdict
from collections import Counter
from flask import Flask, request, send_file, jsonify, abort
from wordcloud import WordCloud
from datetime import datetime
from LeIA import SentimentIntensityAnalyzer
from flask_cors import CORS
import tempfile
import zipfile
import io
import pandas as pd
import matplotlib
import chardet
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

ALLOWED_EXTENSIONS = {'txt', 'zip', 'zipfile'}
ALLOWED_HOSTS = ["https://analisegrupal.com.br", "https://api.analisegrupal.com.br"]

openai.api_key = os.getenv("CHATGPT_ABECMED_APIKEY")

nltk.download('stopwords')
nltk.download('wordnet')

_stop_words = set(stopwords.words('portuguese'))
lemma = WordNetLemmatizer()

app = Flask(__name__)
CORS(app)

# s = SentimentIntensityAnalyzer() # non portuguese

# List of common Portuguese stop words
# TODO improve
portuguese_stop_words = [
    "omitted", "Media", "Mas", "mas", "q", "O", "E", "mais", "omitted>", "<Media", "é", "pra", "eu", "tá", "http", "https", "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "com", "como", "da", "das", "de", 
    "dela", "delas", "dele", "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", 
    "eram", "essa", "essas", "esse", "esses", "esta", "estamos", "estas", "estava", "estavam", "este", "esteja", "estejam", 
    "estejamos", "estes", "esteve", "estive", "estivemos", "estiver", "estivera", "estiveram", "estiverem", "estivermos", 
    "estivesse", "estivessem", "estivéramos", "estivéssemos", "estou", "eu", "foi", "fomos", "for", "fora", "foram", "forem", 
    "formos", "fosse", "fossem", "fui", "fôramos", "fôssemos", "haja", "hajam", "hajamos", "havemos", "hei", "houve", 
    "houvemos", "houver", "houvera", "houveram", "houverei", "houverem", "houveremos", "houveria", "houveriam", "houvermos", 
    "houverá", "houverão", "houveríamos", "houvesse", "houvessem", "houvéramos", "houvéssemos", "isso", "isto", "já", "lhe", 
    "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nem", "no", "nos", "nossa", 
    "nossas", "nosso", "nossos", "num", "numa", "não", "nós", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", 
    "qual", "quando", "que", "quem", "se", "seja", "sejam", "sejamos", "sem", "serei", "seremos", "seria", "seriam", "será", 
    "serão", "seríamos", "seu", "seus", "somos", "sou", "sua", "suas", "são", "só", "também", "te", "tem", "temos", "tenha", 
    "tenham", "tenhamos", "tenho", "terei", "teremos", "teria", "teriam", "terá", "terão", "teríamos", "teu", "teus", "teve", 
    "tinha", "tinham", "tive", "tivemos", "tiver", "tivera", "tiveram", "tiverem", "tivermos", "tivesse", "tivessem", 
    "tivéramos", "tivéssemos", "tu", "tua", "tuas", "tém", "tínhamos", "um", "uma", "você", "vocês", "vos", "à", "às", "éramos"
]

# Define keywords for each topic
# TODO improve
politics_keywords = {
    "presidente", "eleição", "governo", "voto", "partido", "política", 
    "senador", "deputado", "campanha", "ministro", "estado", "município", 
    "lei", "congresso", "constituição", "oposição", "reforma", "impeachment", 
    "legislação", "corrupção", "tribunal", "justiça", "candidato", "urna", "plebiscito"
}

# TODO improve
religion_keywords = {
    "deus", "igreja", "rezar", "religião", "bíblia", "santo", "padre", 
    "oração", "fé", "espírito", "evangélico", "católico", "papa", 
    "culto", "missa", "milagre", "paróquia", "bispo", "pastor", "salmos", "crença"
}

# TODO improve
soccer_keywords = {
    "golaco" ,"golaço", "gol", "jogo", "time", "placar", "futebol", "partida", "campeonato", 
    "jogador", "torcida", "estádio", "seleção", "treinador", "escalação", 
    "cartão", "falta", "pênalti", "liga", "derrota", "vitória", "empate", "goleiro", "taça"
}

# TODO improve
sex_pornography_keywords = {
    "brotheragem", "holandês", "holandes", "puta", "gostosa", "safada", "pornografia", "nu", "nudez", "sensual", "erotismo", 
    "pornô", "fetiche", "prostituição", "adulto", "orgasmo", "lubrificante", 
    "preservativo", "camisinha", "vibrador", "strip", "lingerie", "sedução"
}

alcohol_keywords = {
    "cerveja", "vinho", "whisky", "vodka", "caipirinha", "tequila", 
    "bebida", "bar", "pub", "balada", "festa", "drink", "drinks", 
    "porre", "ressaca", "brinde", "saideira", "happy hour", "destilado", 
    "fermentado", "alcoólico", "cocktail", "mixologia", "bartender", 
    "chopp", "bebado", "bebum", "etilico"
}


# TODO improve
remove_words = [
    "omitted", "Media", "Mas", "mas", "q", "O", "E", "mais", "omitted>", "<Media", "media", "Media", "http", 
    "https", "figurinha omitida", "imagem ocultada", "oculto>", "mídia", "[]", "<Aruivo", "apagada", "Mensagem",
    "<", "editada>", ">", "message", "Message", "deleted", "Deleted", "This", "this", "file attached", "attached",
    "Arquivo oculto", "Arquivo", "oculto", "vídeo omitido", "imagem ocultada", "ocultada", "imagem", "ocultado áudio",
    "ocultado", "áudio", "ocultado audio", "omitida", "figurinha", "video", "omitido", "anexado", "sticker", "image", 
    "imagem", "vídeo", "foto", "edited", "igshid", "open spotify", "open", "spotify", "gmail", "foto", "youtube", "instagram"
]

def preprocess(text):
    # Convert both lists to sets and then union them
    temp_set = set(remove_words).union(set(portuguese_stop_words))
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Filter out tokens in the stop words set and that are greater than 3 characters
    tokens = [word for word in tokens if word not in temp_set and len(word) > 3]
    
    # Lemmatize the remaining tokens
    tokens = [lemma.lemmatize(word) for word in tokens]
    
    return tokens

def build_lda_model(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary


def call_openai_api(prompt):
    api_key = os.getenv('CHATGPT_ABECMED_APIKEY')
    if not api_key:
        logging.error("No API key found. Please set the CHATGPT_ABECMED_APIKEY environment variable.")
        raise ValueError('No API key found. Please set the CHATGPT_ABECMED_APIKEY environment variable.')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'prompt': prompt,
        'max_tokens': 150,
        'temperature': 0.5,
    }

    response = requests.post('https://api.openai.com/v1/engines/text-davinci-003/completions', json=data, headers=headers)
    if response.status_code == 200:
        logging.info("OpenAI API call successful.")
        return response.json()["choices"][0]["text"].strip()
    else:
        logging.error(f"Failed to call OpenAI API: {response.status_code} {response.text}")
        return None

def generate_pattern(unmatched_line, patterns, depth=0, max_depth=2):
    if depth > max_depth:
        logging.warning("Maximum recursion depth reached without finding a match.")
        return None

    logging.info("Attempting to match the line with existing patterns.")
    for pattern in patterns:
        if re.match(pattern, unmatched_line):
            logging.info(f"Existing pattern matched: {pattern.pattern}")
            return pattern.pattern

    logging.info("No existing pattern matched. Asking OpenAI API for a suggestion.")
    prompt = f"Generate a regex pattern that matches the following line:\n'{unmatched_line}'\n\nExisting patterns:\n"
    for p in patterns:
        prompt += f"{p.pattern}\n"
    prompt += "\nSuggested pattern:"

    suggested_pattern = call_openai_api(prompt)
    if suggested_pattern:
        logging.info(f"OpenAI API suggested a pattern: {suggested_pattern}")
        try:
            compiled_pattern = re.compile(suggested_pattern, re.IGNORECASE)
            if compiled_pattern.match(unmatched_line):
                logging.info(f"Suggested pattern successfully matched the line: {suggested_pattern}")
                return suggested_pattern
            else:
                logging.warning(f"Suggested pattern did not match the line: {suggested_pattern}")
                # Recurse with increased depth
                return generate_pattern(unmatched_line, patterns, depth=depth + 1)
        except re.error as e:
            logging.error(f"Generated pattern is not a valid regex: {e}")
            # Recurse with increased depth
            return generate_pattern(unmatched_line, patterns, depth=depth + 1)
    else:
        logging.error("OpenAI API did not provide a suggested pattern.")
        return None

def format_datetime(date, time):
    # Split the date into day, month, and year
    day, month, year = map(int, date.split('/'))
    
    # Ensure the year is in two-digit format
    year = year % 100
    
    # Extract hour, minute, and possibly second from time
    time_parts = time.split(':')
    hour, minute = map(int, time_parts[:2])
    
    # Extract AM or PM if present
    am_pm = time_parts[2].split()[1] if len(time_parts) > 2 and ('AM' in time_parts[2] or 'PM' in time_parts[2]) else None
    if not am_pm and '\u202F' in time:  # Special case with the narrow no-break space
        am_pm = time_parts[2]
    
    # If AM or PM is not provided, make an assumption based on the hour
    if not am_pm:
        am_pm = "AM" if hour < 12 else "PM"
    
    # Format date and time
    formatted_date = f"{day:02}/{month:02}/{year:02}"
    formatted_time = f"{hour:02}:{minute:02} {am_pm}"
    
    return f"{formatted_date}, {formatted_time}"

def preprocess_content_new(content, words_to_remove=[]):
    extracted_content = []
    patterns = [
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?)\] (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?) ?(AM|PM)? - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2})\u202F(AM|PM) - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) \| (.*?) \| (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) \| (.*?) \- (.*)', re.IGNORECASE)
    ]

    for line in content:
        line = line.replace('"', "'")
        print(line)
        matched = False
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                date, time = match.groups()[0], match.groups()[1]
                
                # Format the datetime to ensure it's in the desired format
                formatted_datetime = format_datetime(date, time)
                
                person = match.groups()[-2]
                message_content = match.groups()[-1]

                _str = formatted_datetime + ' - ' + person.strip() + ": " + message_content.strip()
                for word in words_to_remove:
                    _str = _str.replace(word, "")
                extracted_content.append(_str)

                matched = True
                break
        
        if not matched:
            # No existing pattern matched, try to generate a new one
            new_pattern = generate_pattern(line, patterns)
            if new_pattern:
                # Compile the new pattern and add it to the list
                compiled_new_pattern = re.compile(new_pattern, re.IGNORECASE)
                patterns.append(compiled_new_pattern)
                
                # Now that we have a new pattern, retry matching the line
                match = compiled_new_pattern.search(line)
                if match:

                    date, time = match.groups()[0], match.groups()[1]
                    # Format the datetime to ensure it's in the desired format
                    formatted_datetime = format_datetime(date, time)
                    
                    person = match.groups()[-2]
                    message_content = match.groups()[-1]

                    _str = formatted_datetime + ' - ' + person.strip() + ": " + message_content.strip()
                    for word in words_to_remove:
                        _str = _str.replace(word, "")
                    extracted_content.append(_str)
                    break
                else:
                    logging.warning(f"New pattern did not match the line: {line}")
            else:
                logging.error(f"Could not generate a new pattern for the line: {line}")

    return extracted_content


def preprocess_content(content, words_to_remove=[]):
    extracted_content = []

    # Define patterns to handle various date-time and message structures
    patterns = [
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?)\] (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?) ?(AM|PM)? - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2})\u202F(AM|PM) - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?): \u200e?(.*)', re.IGNORECASE)
    ]
    
    for line in content:
        line = line.replace('"', "'")
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                date, time = match.groups()[0], match.groups()[1]
                
                # Format the datetime to ensure it's in the desired format
                formatted_datetime = format_datetime(date, time)
                
                person = match.groups()[-2]
                message_content = match.groups()[-1]

                # extracted_content.append((formatted_datetime, person.strip(), message_content.strip()))
                _str = formatted_datetime + ' - ' + person.strip() + ": " + message_content.strip()
                # print(_str)
                for word in words_to_remove:
                    _str = _str.replace(word, "")
                extracted_content.append(_str)
                break

    return extracted_content

def extract_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_file(file_stream):
    try:
        return file_stream.read().decode('utf-8').splitlines()
    except UnicodeDecodeError:
        file_stream.seek(0)  # Go back to the beginning of the file
        raw_data = file_stream.read()
        encoding = chardet.detect(raw_data)['encoding']
        file_stream.seek(0)  # Go back to the beginning of the file again
        return file_stream.read().decode(encoding).splitlines()
    except Exception as e:
        print(f"An error occurred in decode_file: {e}")
        raise  # Re-raise the exception to be handled by the calling function

def process_file(request):
    if 'file' not in request.files:
        logging.debug("No file part in request")
        return None, 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        logging.debug("No selected file")
        return None, 'No selected file', 400
    if not allowed_file(file.filename):
        logging.debug("File type not allowed")
        return None, 'File type not allowed', 400

    try:
        if zipfile.is_zipfile(file):
            with zipfile.ZipFile(file) as z:
                txt_file = next((f for f in z.namelist() if f.lower().endswith('.txt')), None)
                if txt_file is None:
                    return None, 'No txt file found in the zip', 400
                with z.open(txt_file) as f:
                    content = preprocess_content(decode_file(f))
        else:
            file.seek(0)
            _content = file.read().decode('utf-8', errors='replace').splitlines()
            content = preprocess_content(_content)
        return content, None, None
    except Exception as e:
        logging.exception("Error processing file")
        return None, str(e), 500

def extract_and_process_sentiments(content):
    """
    Extracts timestamps and messages from the content and computes sentiment scores.
    """
    message_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}) - .*?: (.*)")
    extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

    if not extracted_data:
        return None, "No valid data extracted"

    timestamps, messages = zip(*extracted_data)

    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, errors='coerce', format='%m/%d/%y, %I:%M %p'),
        'sentiment': sentiments
    }).dropna()

    df.set_index('timestamp', inplace=True)
    df = df.resample('D').mean().fillna(0)

    return df, None


def extract_and_analyze_sentiment(content):
    """
    Extracts messages and analyzes sentiment to calculate average sentiment per person.
    """
    message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?): (.*)")
    extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

    senders, messages = zip(*extracted_data)

    # Compute sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

    # Calculate average sentiment per person
    sender_sentiments = defaultdict(list)
    for sender, sentiment in zip(senders, sentiments):
        sender_sentiments[sender].append(sentiment)

    avg_sentiments = {sender: sum(vals)/len(vals) for sender, vals in sender_sentiments.items()}

    return avg_sentiments


def identify_significant_changes(monthly_avg):
    """
    Identifies significant changes in monthly average sentiment.
    """
    changes_summary = ""
    previous_month_sentiment = None

    for month, row in monthly_avg.iterrows():
        if previous_month_sentiment is not None:
            change = row['sentiment'] - previous_month_sentiment
            if abs(change) > 0.014:  # Assuming a change greater than 0.014 is significant; adjust as needed
                change_type = "increase" if change > 0 else "decrease"
                changes_summary += (f"- There was a significant {change_type} in sentiment in {month.strftime('%B %Y')} "
                                    f"(change: {change:.2f}).\n")
        previous_month_sentiment = row['sentiment']

    if not changes_summary:
        changes_summary = "No significant month-to-month changes were detected."

    return changes_summary


def construct_prompt_for_sentiment_analysis(df, analysis_type):
    """
    Constructs a human-like prompt from the sentiment over time data.
    Adjusts the level of detail based on the number of months in the dataset.
    """
    # Resample the data to get monthly averages
    monthly_avg = df.resample('M').mean()

    num_months = len(monthly_avg)
    summary = "The data shows how sentiment scores vary over time in a WhatsApp group chat, summarized by month. "

    if num_months > 72:
        # For larger datasets, focus on significant changes or patterns
        significant_changes = identify_significant_changes(monthly_avg)
        summary += "Here are some significant changes and patterns:\n" + significant_changes
    else:
        # For smaller datasets, provide a month-by-month summary
        for month, row in monthly_avg.iterrows():
            sentiment_description = "positive" if row['sentiment'] > 0 else "negative" if row['sentiment'] < 0 else "neutral"
            summary += f"- For {month.strftime('%B %Y')}, the average sentiment was {sentiment_description}, with a score of {row['sentiment']:.2f}.\n"

    # Tailor the prompt based on the analysis type
    if analysis_type == 'technical':
        prompt = summary + "Please provide a technical explanation of what these trends might suggest about the group dynamics."
    elif analysis_type == 'fun':
        prompt = summary + "Let's make this fun. Provide a creative and lighthearted explanation of these trends."
    elif analysis_type == 'zoeira':
        prompt = summary + "Please provide a sarcastic and humorous analysis of the data. Remember, 'The zueira never ends!'"
    prompt += " Em português, por favor."

    return prompt


def construct_shorter_prompt_from_data_avg_sentiments(avg_sentiments, type):
    """
    Constructs a human-like prompt from the sentiment analysis data, using two decimal places for sentiment scores,
    focusing on the top and bottom 7 sentiments.
    """
    # Sort the sentiments from highest to lowest
    sorted_sentiments = sorted(avg_sentiments.items(), key=lambda x: x[1], reverse=True)
    top_sentiments = sorted_sentiments[:12]  # Top 12 sentiments
    bottom_sentiments = sorted_sentiments[-12:]  # Bottom 12 sentiments

    prompt = "The data shows the average sentiment scores per person in a WhatsApp group chat. Here are some key insights:\n"

    # Construct prompt for top sentiments
    for sender, sentiment in top_sentiments:
        sentiment_description = "positive" if sentiment > 0 else "neutral" if sentiment == 0 else "negative"
        prompt += f"- Sentiment for {sender} is {sentiment_description}, with an avg score of {sentiment:.2f}.\n"

    # Construct prompt for bottom sentiments
    for sender, sentiment in bottom_sentiments:
        sentiment_description = "positive" if sentiment > 0 else "neutral" if sentiment == 0 else "negative"
        prompt += f"- Sentiment for {sender} is {sentiment_description}, with an avg score of {sentiment:.2f}.\n"
    
    # Add additional message based on type
    if type == 'technical':
        prompt += "Please provide a technical explanation of what this might suggest about the group dynamics."
    elif type == 'fun':
        prompt += "Let's make this fun and not serious. Provide a brief, but creative and fun explanation of what this might suggest about the group dynamics."
    elif type == 'zoeira':
        prompt += "Please provide a sarcastic fun analysis of the data. Be intelligent, bold, you can get creative here (not super long please). There's a saying in Brazil: 'The zueira never ends!' Take this saying seriously."
    prompt += " Em português, por favor."
    
    return prompt


def construct_prompt_from_data_avg_sentiments(avg_sentiments, type):
    """
    Constructs a human-like prompt from the sentiment analysis data, using two decimal places for sentiment scores.
    """
    prompt = "The data shows the average sentiment scores per person in a WhatsApp group chat. Here are some key insights:\n"
    for sender, sentiment in avg_sentiments.items():
        sentiment_description = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        prompt += f"- Sentiment for {sender} is {sentiment_description}, with an average score of {sentiment:.2f}.\n"
    
    if type == 'technical':
        prompt += "Please provide a technical explanation of what this might suggest about the group dynamics."
    elif type == 'fun':
        prompt += "Let's make this fun and not serious. Provide a brief, but creative and fun explanation of what this might suggest about the group dynamics."
    elif type == 'zoeira':
        prompt += "Please provide a sarcastic fun analysis of the data. Be intelligent, bold, you can get creative here (not super long please). Theres a saying in brazil; 'The zueira never ends!' take this saying seriously."
    prompt += " Em português, por favor."
    
    return prompt


def call_chatgpt_api(prompt, model_name, length=150, temperature=0):
    if not openai.api_key:
        logging.error("No API key found. Please set the CHATGPT_ABECMED_APIKEY environment variable.")
        raise ValueError('No API key found. Please set the CHATGPT_ABECMED_APIKEY environment variable.')

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "This is a message."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=length,
            temperature=temperature
        )
        if response:
            logging.info("OpenAI API call successful.")
            return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        logging.error(f"An error occurred while calling OpenAI API: {e}")
        return None


@app.before_request
def limit_referer():
    if os.environ.get('FLASK_ENV') == 'production':
        allowed_referer = "https://analisegrupal.com.br/"
        
        referer = request.headers.get('Referer')

        if referer and not referer.startswith(allowed_referer):
            logging.debug(f'Forbiden referer: {referer} - 403')
            abort(403)  # Forbidden


@app.route('/')
def home():
    return jsonify({"message": "Hello, allowed host!"})


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify(success=True), 200


@app.route('/upload_to_imgur', methods=['POST'])
def upload_to_imgur():
    imgur_client_id = os.getenv('IMGUR_CLIENT_ID')
    if not imgur_client_id:
        logging.error('IMGUR_CLIENT_ID environment variable not set.')
        # Attempt Cloudinary upload instead
        return upload_to_cloudinary()

    if 'image' not in request.files:
        logging.error('No image part in request.')
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        logging.error('No selected image.')
        return jsonify({'error': 'No selected image'}), 400

    try:
        # Read the image and encode it in base64
        image_b64 = base64.b64encode(file.read()).decode('utf-8')

        headers = {'Authorization': f'Client-ID {imgur_client_id}'}
        data = {'image': image_b64, 'type': 'base64'}

        # Send the POST request to Imgur
        response = requests.post('https://api.imgur.com/3/image', headers=headers, data=data)

        if response.status_code == 200:
            # Extract the link from the response data
            link = response.json()['data']['link']
            logging.info(f'Image successfully uploaded to Imgur: {link}')
            return jsonify({'link': link})
        else:
            logging.warning(f'Imgur upload failed, attempting Cloudinary upload.')
            # Attempt Cloudinary upload instead
            return upload_to_cloudinary()

    except requests.RequestException as e:
        logging.exception('Request failed with Imgur.')
        # Attempt Cloudinary upload instead
        return upload_to_cloudinary()
    except Exception as e:
        logging.exception('An unexpected error occurred.')
        return jsonify({'error': 'An unexpected error occurred', 'message': str(e)}), 500


def upload_to_cloudinary():
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_UPLOAD_PRESET = os.getenv('CLOUDINARY_UPLOAD_PRESET')

    # Check if all required Cloudinary configuration variables are set
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_UPLOAD_PRESET]):
        logging.error('Cloudinary environment variables not set.')
        return jsonify({'error': 'Image upload service not available'}), 500

    # Get the file from the request
    file = request.files['image']
    if file.filename == '':
        logging.error('No selected image.')
        return jsonify({'error': 'No selected image'}), 400

    try:
        # Ensure the file pointer is at the start
        file.seek(0)
        # Read the image data
        image_data = file.read()

        # Send the POST request to Cloudinary
        response = requests.post(
            f'https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload',
            files={'file': (file.filename, image_data)},  # Send file with filename
            data={'upload_preset': CLOUDINARY_UPLOAD_PRESET}
        )

        # Check if the upload was successful
        if response.status_code == 200:
            link = response.json().get('secure_url')
            logging.info(f'Image successfully uploaded to Cloudinary: {link}')
            return jsonify({'link': link})
        else:
            logging.error(f'Cloudinary upload failed with status code {response.status_code}: {response.content}')
            return jsonify({'error': 'Upload failed', 'response': response.json()}), response.status_code

    except requests.RequestException as e:
        logging.exception('Request failed with Cloudinary.')
        return jsonify({'error': 'Upload failed', 'message': str(e)}), 500
    except Exception as e:
        logging.exception('An unexpected error occurred with Cloudinary.')
        return jsonify({'error': 'An unexpected error occurred', 'message': str(e)}), 500



def is_drinking_invitation(message):
    """
    Check if a message contains an invitation to drink.

    Parameters:
    - message (str): The text message to be analyzed.

    Returns:
    - bool: True if the message is an invitation to drink, False otherwise.
    """
    # Keywords that could indicate an invitation
    invitations = ["vamo", "vamos", "bora", "partiu", "simba", "simbora", "e aí", "e ai"]
    
    # Phrases that suggest the action is drinking
    drinking_phrases = ["beber", "tomar uma", "porre", "bebericar"]
    
    # Check if any combination of invitation and drinking phrase is in the message
    return any(invite in message for invite in invitations) and any(drink in message for drink in drinking_phrases)


@app.route('/whatsapp/message/topic_modeling', methods=['POST'])
def topic_modeling():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [msg for msg in messages if msg is not None]

        preprocessed_messages = [preprocess(message) for message in messages]
        lda_model, corpus, dictionary = build_lda_model(preprocessed_messages, num_topics=5)

        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

        # This will generate HTML. For a Flask route, you might want to return this HTML.
        # Or, if you prefer, you can save this to a file and return the file path.
        html = pyLDAvis.prepared_data_to_html(vis_data)

        return html  # This will directly return the visualization as an HTML page.

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/avg_sentiment_per_person', methods=['POST'])
def plot_avg_sentiment_per_person():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # print(f"@@@@@@@ Content preview: {content[:5]}")
        avg_sentiments = extract_and_analyze_sentiment(content)

        if len(avg_sentiments) == 0:  # Check if the DataFrame is empty
            return "No data available for plotting - avg_sentiments.empty", 400

        N = 20  # Number of senders with the highest sentiment scores to display
        M = 20  # Number of senders with the lowest sentiment scores to display

        # Sort the dictionary by average sentiment
        sorted_avg_sentiments = dict(sorted(avg_sentiments.items(), key=lambda item: item[1]))

        # Extract the top N and bottom M senders
        top_senders = dict(list(sorted_avg_sentiments.items())[-N:])
        bottom_senders = dict(list(sorted_avg_sentiments.items())[:M])

        # Merge the two dictionaries
        combined_senders = {**bottom_senders, **top_senders}

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.bar(combined_senders.keys(), combined_senders.values(), color='dodgerblue')
        plt.title("Top and Bottom Senders by Average Sentiment")
        plt.ylabel("Average Sentiment Score")
        plt.xlabel("Sender")
        plt.xticks(rotation=45, ha='right')

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.tight_layout()
        plt.savefig(temp_file.name, format='png')
        plt.close()

        return send_file(temp_file.name, mimetype='image/png')
    
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/avg_sentiment_per_person/analyse', methods=['POST'])
def analyse_avg_sentiment_per_person():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:
        data = json.loads(request.form.get('data', '{}'))
        analysis_type = data.get('type', 'technical')  # Default to 'technical' if not provided
        temperature = data.get('temperature', 0)  # Default to 0 if not provided
        
        # print(f"@@@@@@@ Content preview: {content[:5]}")
        avg_sentiments = extract_and_analyze_sentiment(content)

        if len(avg_sentiments) == 0:
            return "No data available for analysis - avg_sentiments.empty", 400

        # print(len(avg_sentiments))
        # print("****")

        if len(avg_sentiments) > 40:
            prompt = construct_shorter_prompt_from_data_avg_sentiments(avg_sentiments, analysis_type)
        else:
            # Construct a prompt for the ChatGPT API
            prompt = construct_prompt_from_data_avg_sentiments(avg_sentiments, analysis_type)

        # print(prompt)

        length = 350 if analysis_type == 'technical' else 400 if analysis_type == 'fun' else 500 if analysis_type == 'zoeira' else 300
        chatgpt_response = call_chatgpt_api(prompt, "gpt-3.5-turbo", length, temperature)

        # For the sake of this example, we're just returning the mock response directly
        return jsonify(chatgpt_response)
    
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/length_over_time', methods=['POST'])
def plot_message_length_over_time():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # print(f"Content preview: {content[:5]}")
        
        # Regular expression to extract timestamp and messages
        message_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}) - .*?: (.*)")
        extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

        # print(extracted_data[:5])

        timestamps, messages = zip(*extracted_data)

        # Compute message lengths
        message_lengths = [len(msg) for msg in messages]

        df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, errors='coerce', format='%m/%d/%y, %I:%M\u202f%p'),
        'message_length': [len(msg) for msg in messages]
    }).dropna()

        # print(df)

        df.set_index('timestamp', inplace=True)
        df = df.resample('D').mean()

        # print(df)

        if df['message_length'].dropna().empty:
            return "No data available for plotting", 400

        # Plotting
        plt.figure(figsize=(12, 8))
        df['message_length'].plot()
        plt.title("Message Length Over Time")
        plt.ylabel("Average Message Length")
        plt.xlabel("Date")

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/avg_sentiment_per_person/json', methods=['POST'])
def avg_sentiment_per_person_json():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Regular expression to extract timestamp, sender and messages
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?): (.*)")
        extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

        senders, messages = zip(*extracted_data)

        # Compute sentiment scores using LeIA
        analyzer = SentimentIntensityAnalyzer()
        sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

        # Calculate average sentiment per person
        sender_sentiments = defaultdict(list)
        for sender, sentiment in zip(senders, sentiments):
            sender_sentiments[sender].append(sentiment)

        avg_sentiments = {sender: sum(vals)/len(vals) for sender, vals in sender_sentiments.items()}

        # Return the average sentiments as a JSON response
        return jsonify(avg_sentiments)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/sentiment_over_time', methods=['POST'])
def plot_sentiment_over_time():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    df, error_message = extract_and_process_sentiments(content)
    if df is None:
        return jsonify({'status': 'error', 'message': error_message}), 400

    try:
        
        # Regular expression to extract timestamp and messages
        message_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}) - .*?: (.*)")
        extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

        timestamps, messages = zip(*extracted_data)

        # print(f"Timestamps count: {len(timestamps)}")

        # Compute sentiment scores
        analyzer = SentimentIntensityAnalyzer()
        sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

        # print(f"Sentiments count: {len(sentiments)}")

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, errors='coerce', format='%m/%d/%y, %I:%M %p'),
            'sentiment': sentiments
        }).dropna()

        df.set_index('timestamp', inplace=True)
        df = df.resample('D').mean().fillna(0)

        # print(df.head())  # Print the first few rows of the DataFrame for debugging

        if df.empty:  # Check if the DataFrame is empty
            return "No data available for plotting", 400

        # Plotting
        plt.figure(figsize=(12, 8))
        df['sentiment'].plot()
        plt.title("Sentiment Over Time")
        plt.ylabel("Sentiment Score")
        plt.xlabel("Date")

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/sentiment_over_time/analyse', methods=['POST'])
def analyse_sentiment_over_time():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:
        data = json.loads(request.form.get('data', '{}'))
        analysis_type = data.get('type', 'technical')  # Default to 'technical' if not provided
        temperature = data.get('temperature', 0)  # Default to 0 if not provided

        df, error_message = extract_and_process_sentiments(content)
        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': error_message or "No data available for analysis"}), 400

        # Construct a prompt for the ChatGPT API based on the sentiment data
        prompt = construct_prompt_for_sentiment_analysis(df, analysis_type)

        print(prompt)

        length = 350 if analysis_type == 'technical' else 400 if analysis_type == 'fun' else 500 if analysis_type == 'zoeira' else 300
        chatgpt_response = call_chatgpt_api(prompt, "gpt-3.5-turbo", length, temperature)

        # Returning the ChatGPT API response
        return jsonify(chatgpt_response)
    
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/peak_response_time', methods=['POST'])
def analyze_peak_response_time():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # print(content[:10])  # Print the first 10 lines

        # Update the regex pattern
        timestamp_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}(?:\s?[APMapm]{2})?) - (.*?):")

        # Lists to store the extracted timestamps and senders
        timestamps = []
        senders = []

        # Function to correct invalid time format
        def correct_time_format(timestamp_str):
            time_part = timestamp_str.split(", ")[1].split(" ")[0]
            hour, minute = map(int, time_part.split(":"))
            
            if "AM" in timestamp_str or "PM" in timestamp_str:
                # Adjust for midnight or noon
                if hour == 0:
                    timestamp_str = timestamp_str.replace("00:", "12:")
                # Remove PM/AM for 24-hour format
                if hour >= 12:
                    timestamp_str = timestamp_str.replace(" PM", "")
                else:
                    timestamp_str = timestamp_str.replace(" AM", "")
            return timestamp_str

        # Use the modified function in the previous code
        for line in content:
            match = timestamp_pattern.search(line)
            if match:
                timestamp_str, sender = match.groups()
                timestamp_str = correct_time_format(timestamp_str)
                # Adjust datetime parsing based on whether AM/PM exists in the string
                if "AM" in timestamp_str or "PM" in timestamp_str:
                    timestamp_format = "%d/%m/%y, %I:%M %p"
                else:
                    timestamp_format = "%d/%m/%y, %H:%M"
                try:
                    timestamp = datetime.strptime(timestamp_str, timestamp_format)
                except ValueError:
                    timestamp_format = "%m/%d/%y, %H:%M"  # fallback to month/day/year
                    timestamp = datetime.strptime(timestamp_str, timestamp_format)
                timestamps.append(timestamp)
                senders.append(sender)

        # print("timestamps: ", str(len(timestamps)))
        # print("senders: ", str(len(senders)))

        # Create a DataFrame from the extracted data
        df = pd.DataFrame({'Timestamp': timestamps, 'Sender': senders})

        # Extract the hour from each timestamp
        df['Hour'] = df['Timestamp'].dt.hour

        # Calculate the average number of messages sent per hour
        avg_hourly_messages = df.groupby('Hour').size()

        # Calculate the sender who sent the most messages for each hour
        dominant_sender = df.groupby('Hour')['Sender'].agg(lambda x: x.value_counts().idxmax())

        # Plotting
        plt.figure(figsize=(12, 7))
        avg_hourly_messages.plot(kind='bar', color='dodgerblue')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Messages')
        plt.title('Average Messages Per Hour and Dominant Sender')
        
        # Annotate bars with dominant sender's name
        for idx, value in enumerate(avg_hourly_messages):
            plt.text(idx, value + 0.5, dominant_sender.iloc[idx], ha='center', rotation=90, fontsize=8)

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.tight_layout()
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/peak_response_time/json', methods=['POST'])
def analyze_peak_response_time_json():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the timestamp and sender's name
        timestamp_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\u202f[APMapm]{2}) - (.*?):")

        # Lists to store the extracted timestamps and senders
        timestamps = []
        senders = []

        # Iterate over each line to extract timestamps and sender names
        for line in content:
            match = timestamp_pattern.search(line)
            if match:
                timestamp_str, sender = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%m/%d/%y, %I:%M\u202f%p")
                timestamps.append(timestamp)
                senders.append(sender)

        # Create a DataFrame from the extracted data
        df = pd.DataFrame({'Timestamp': timestamps, 'Sender': senders})

        # Extract the hour from each timestamp
        df['Hour'] = df['Timestamp'].dt.hour

        # Calculate the average number of messages sent per hour
        avg_hourly_messages = df.groupby('Hour').size()

        # Calculate the sender who sent the most messages for each hour
        dominant_sender = df.groupby('Hour')['Sender'].agg(lambda x: x.value_counts().idxmax())

        # Return the results as a JSON response
        result = {
            'AverageMessagesPerHour': avg_hourly_messages.to_dict(),
            'DominantSenderPerHour': dominant_sender.to_dict()
        }
        return jsonify(result)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/activity_heatmap', methods=['POST'])
def activity_heatmap():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract date and time details
        date_time_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2})\s[APMapm]{2}")

        # Extract date and time details
        date_times = [date_time_pattern.search(line).groups() for line in content if date_time_pattern.search(line)]
        dates, times = zip(*date_times)

        # Convert to pandas datetime format for easier manipulation
        date_times = pd.to_datetime([' '.join(item) for item in zip(dates, times)], errors='coerce')
        df = pd.DataFrame({'datetime': date_times})

        # Extract day of week and hour from the datetime
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour'] = df['datetime'].dt.hour

        # Create a pivot table for the heatmap
        heatmap_data = df.pivot_table(index='day_of_week', columns='hour', aggfunc='size', fill_value=0)

        # Define the order of days for the y-axis
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)

        # Plotting the heatmap
        plt.figure(figsize=(14, 7))
        sns.heatmap(heatmap_data, cmap='YlGnBu', cbar_kws={'label': 'Number of Messages'})
        plt.title("Activity Heatmap (Messages over Time)")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Day of the Week")

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/user_activity_over_time', methods=['POST'])
def user_activity_over_time():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the date and participant names
        message_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}), \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?): .*")
        
        # Extract dates and names
        dates_names = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]
        
        # Convert to DataFrame
        df = pd.DataFrame(dates_names, columns=['Date', 'Name'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Group by date and name and count messages
        grouped = df.groupby(['Date', 'Name']).size().unstack(fill_value=0)
        
        # Plotting
        plt.figure(figsize=(15, 8))
        for column in grouped.columns:
            plt.plot(grouped.index, grouped[column], label=column)
        plt.title("User Activity Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        plt.legend(loc="upper right")
        
        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/top_emojis_json/<int:top_n>', methods=['POST'])
def get_top_emojis_json(top_n=10):
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the message content
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [msg for msg in messages if msg is not None]

        # Count the emojis
        all_emojis = [emoji for message in messages for emoji in extract_emojis(message)]
        emoji_counts = Counter(all_emojis)

        print(emoji_counts)

        # Prepare data for output
        sorted_emoji_counts = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Return the sorted emoji count as JSON
        return jsonify(sorted_emoji_counts)

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Conversational Turn Analysis
# In a group chat, it's interesting to see how often the conversation "turns" 
# to a new person. For example, if Person A sends 5 messages in a row, then Person B 
# sends 2 messages, and then Person C sends 1 message, there were 3 conversational turns. 
# This analysis can give insights into the flow of the conversation, indicating whether 
# it's dominated by long monologues or if it's more dynamic with many participants 
# chiming in frequently.
@app.route('/whatsapp/message/conversational_turns', methods=['POST'])
def plot_conversational_turns():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the sender of each message
        sender_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?):")
        senders = [sender_pattern.search(line).group(1) if sender_pattern.search(line) else None for line in content]
        senders = [sender for sender in senders if sender is not None]
     
        # Count the conversational turns
        turn_counts = defaultdict(int)
        previous_sender = None
        for sender in senders:
            if sender != previous_sender:
                turn_counts[sender] += 1
            previous_sender = sender

        N = 20  # Number of senders with the most conversational turns to display
        M = 5  # Number of senders with the fewest conversational turns to display

        # Sort the dictionary by number of turns
        sorted_turn_counts = sorted(turn_counts.items(), key=lambda item: item[1])

        # Extract the top N and bottom M senders
        top_senders = dict(list(sorted_turn_counts)[-N:])
        bottom_senders = dict(list(sorted_turn_counts)[:M])

        # Merge the two dictionaries
        combined_senders = {**bottom_senders, **top_senders}

        # Plotting the data
        plt.figure(figsize=(12, 8))
        names, counts = zip(*combined_senders.items())
        plt.barh(names, counts, color='mediumseagreen')
        plt.xlabel('Number of Turns')
        plt.ylabel('Names')
        plt.title("Top and Bottom Senders by Conversational Turns")
        plt.gca().invert_yaxis()

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# WEIRD
@app.route('/whatsapp/message/mention_analysis', methods=['POST'])
def mention_analysis():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Regular expression to extract sender and messages
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?): (.*)")
        extracted_data = [(match.group(1), match.group(2)) for line in content if (match := message_pattern.search(line))]

        senders, messages = zip(*extracted_data)
        all_senders = set(senders)

        mention_counts = defaultdict(lambda: defaultdict(int))
        for sender, message in extracted_data:
            for potential_mention in all_senders:
                if potential_mention in message:
                    mention_counts[sender][potential_mention] += 1

        # Preparing data for plotting
        names = list(all_senders)
        mention_matrix = [[mention_counts[sender][mentioned] for mentioned in names] for sender in names]

        # Visualizing the data
        plt.figure(figsize=(12, 8))
        plt.imshow(mention_matrix, cmap='viridis', interpolation='nearest')
        plt.xticks(ticks=range(len(names)), labels=names, rotation=45)
        plt.yticks(ticks=range(len(names)), labels=names)
        plt.colorbar(label="Mention Counts")
        plt.title("Mention Analysis")
        
        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/active_days', methods=['POST'])
def plot_active_days():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the date of each message
        date_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}), \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?:")
        dates = [date_pattern.search(line).group(1) if date_pattern.search(line) else None for line in content]
        dates = [date for date in dates if date is not None]

        # Convert the dates to days of the week
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        days = [days_of_week[pd.to_datetime(date).dayofweek] for date in dates]

        # Count the messages for each day of the week
        day_counts = Counter(days)

        # Prepare data for plotting
        day_labels = days_of_week
        day_values = [day_counts.get(day, 0) for day in day_labels]

        # Plotting the data
        plt.figure(figsize=(12, 7))
        plt.bar(day_labels, day_values, color='lightcoral')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Messages')
        plt.title("Activity Analysis by Day of the Week")

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/topic_percentage', methods=['POST'])
def topic_percentage():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:
        
        # Extract messages from the content
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [msg for msg in messages if msg is not None]

        # Check messages for presence of keywords
        topic_counts = {
            'politics': sum(1 for msg in messages if any(keyword in msg for keyword in politics_keywords)),
            'religion': sum(1 for msg in messages if any(keyword in msg for keyword in religion_keywords)),
            'soccer': sum(1 for msg in messages if any(keyword in msg for keyword in soccer_keywords)),
            'sex & pornography': sum(1 for msg in messages if any(keyword in msg for keyword in sex_pornography_keywords)),
            'alcohol': sum(1 for msg in messages if any(keyword in msg for keyword in alcohol_keywords) or is_drinking_invitation(msg))
        }

        # Convert message counts to percentages
        total_messages = len(messages)
        topic_percentages = {topic: (count/total_messages)*100 for topic, count in topic_counts.items()}

        # Plotting
        plt.figure(figsize=(12, 7))
        plt.bar(topic_percentages.keys(), topic_percentages.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
        plt.ylabel('Percentage of Messages (%)')
        plt.title('Percentage of Messages by Topic')
        
        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/topic_percentage/json', methods=['POST'])
def topic_percentage_json():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Extract messages from the content
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = " ".join([message_pattern.search(line).group(1) if message_pattern.search(line) else "" for line in content])

        # Preprocess the text
        messages = messages.lower()
        tokens = word_tokenize(messages)

        # Remove stopwords
        # stop_words = set(stopwords.words('portuguese'))
        stop_words = _stop_words.union(portuguese_stop_words)
        tokens = [token for token in tokens if token not in stop_words]

        # Count occurrences
        politics_count = sum(token in politics_keywords for token in tokens)
        religion_count = sum(token in religion_keywords for token in tokens)
        soccer_count = sum(token in soccer_keywords for token in tokens)

        total_words = len(tokens)

        # Calculate percentages
        politics_percentage = (politics_count / total_words) * 100
        religion_percentage = (religion_count / total_words) * 100
        soccer_percentage = (soccer_count / total_words) * 100

        return jsonify({
            "politics_percentage": politics_percentage,
            "religion_percentage": religion_percentage,
            "soccer_percentage": soccer_percentage
        })

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/wordfrequency/<int:top_words>', methods=['POST'])
def plot_word_frequency(top_words=20):
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the content of each message
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [message for message in messages if message is not None]

        # stop_words = _stop_words.union(portuguese_stop_words)
        preprocessed_messages = [' '.join(preprocess(message)) for message in messages]

        # Tokenize the messages and count the frequency of each word
        word_freq = Counter()
        for message in preprocessed_messages:
            tokens = message.split()
            word_freq.update(tokens)

        # Extract the top N words for plotting
        top_words_data = word_freq.most_common(top_words)
        words, counts = zip(*top_words_data)

        # Plotting the data
        plt.figure(figsize=(15, 8))
        plt.barh(words, counts, color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.title(f"Top {top_words} Frequently Used Words")
        plt.gca().invert_yaxis()

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/wordcloud', methods=['POST'])
def plot_cleaned_wordcloud():

    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Regular expression to extract the message content
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")


        # Extract message content
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [message for message in messages if message is not None]

        preprocessed_messages = [' '.join(preprocess(message)) for message in messages]

        # # Concatenate all messages
        text = ' '.join(preprocessed_messages)

        # Generate the word cloud
        wordcloud = WordCloud(background_color='white', width=800, height=400, max_words=200).generate(' '.join(preprocessed_messages))

        # Check if the word cloud is effectively empty
        if not wordcloud.words_:
            return "The word cloud is empty, possibly due to too many stopwords or no valid text.", 400

        # Explicitly convert word cloud to image
        wordcloud_image = wordcloud.to_image()

        # Plot the word cloud image
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud_image, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Messages (Cleaned)")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()  # Close the plot

        # Return the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/lenghiest/top/<int:top_contributors>', methods=['POST'])
def plot_lengthiest_messages_pie_chart(top_contributors):
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract participant names and their messages
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?): (.*)")

        # Initialize dictionaries to track total message length and message count for each participant
        total_message_lengths = defaultdict(int)
        message_counts_by_user = defaultdict(int)

        # Iterate over each line to extract participant names and their messages, then update the dictionaries
        for line in content:
            match = message_pattern.search(line)
            if match:
                name, message = match.groups()
                total_message_lengths[name] += len(message)
                message_counts_by_user[name] += 1

        # Compute the average message length for each participant
        average_message_lengths = {name: total_message_lengths[name] / message_counts_by_user[name] for name in total_message_lengths}

        # Sort the participants based on average message length and consider only the top `top_contributors`
        sorted_average_lengths = sorted(average_message_lengths.items(), key=lambda x: x[1], reverse=True)[:top_contributors]

        # Extract names and average lengths for plotting
        top_names = [item[0] for item in sorted_average_lengths]
        top_average_lengths = [item[1] for item in sorted_average_lengths]

        # Create a pie chart to display the top contributors by average message length
        plt.figure(figsize=(12, 8))
        plt.pie(top_average_lengths, labels=top_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f"Top {top_contributors} Contributors by Average Message Length")
        plt.axis('equal')

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/sentiment/distribution', methods=['POST'])
def plot_sentiment_distribution():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract the content of each message
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [message for message in messages if message is not None]

        # Initialize the VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Categorize each message's sentiment
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for message in messages:
            vs = analyzer.polarity_scores(message)
            if vs['compound'] >= 0.05:
                sentiments['positive'] += 1
            elif vs['compound'] <= -0.05:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1

        # Create a pie chart to display the sentiment distribution
        plt.figure(figsize=(12, 8))
        labels = list(sentiments.keys())
        sizes = list(sentiments.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title("Sentiment Distribution of Messages")
        plt.axis('equal')

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/heatmap', methods=['POST'])
def plot_hourly_heatmap():
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Extract hour along with the AM/PM marker using regex
        hour_ampm_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, (\d{1,2}:\d{1,2}\s[APMapm]{2}) - .*?:")
        hours_ampm = [hour_ampm_pattern.search(line).group(1) if hour_ampm_pattern.search(line) else None for line in content]
        hours_ampm = [hour for hour in hours_ampm if hour is not None]

        def convert_to_24_hour(time_str):
            # Strip whitespace for accurate comparison
            time_str = time_str.strip()

            # If the string ends with 'PM' and the hour part is greater than 12, remove the 'PM'
            if time_str.endswith('PM'):
                hour_part = int(time_str.split(":")[0])
                if hour_part >= 12:
                    time_str = time_str.replace(" PM", "")

            # If the string ends with 'AM' or 'PM', convert to 24-hour format
            if time_str.endswith('AM') or time_str.endswith('PM'):
                return pd.to_datetime(time_str).hour
            else:
                # The string is already in 24-hour format, just extract the hour part
                return int(time_str.split(":")[0])


        hours_24 = [convert_to_24_hour(time_str) for time_str in hours_ampm]

        # Count messages for each hour in 24-hour format
        hour_counts_24 = defaultdict(int)
        for hour in hours_24:
            hour_counts_24[hour] += 1

        # Create an array for heatmap
        hour_array_24 = np.zeros(24)
        for hour, count in hour_counts_24.items():
            hour_array_24[hour] = count

        # Plot heatmap
        plt.figure(figsize=(15, 3))
        sns.heatmap([hour_array_24], cmap="YlGnBu", cbar_kws={'label': 'Message Count'}, xticklabels=list(range(24)), yticklabels=[])
        plt.title("Hourly Message Activity (24-hour format)")
        plt.xlabel("Hour of the Day")
        # plt.show()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()  # Close the plot

        # Return the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/whatsapp/message/usercount', methods=['POST'])
def user_message_count():

    # Initialize a dictionary to store the names and their message counts
    message_counts = defaultdict(int)

    # Check if a file was posted
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Regular expression to extract the name pattern from a typical line
        name_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?):")

        # Extract names and count the messages
        for line in content:
            match = name_pattern.search(line)
            if match:
                name = match.group(1)
                message_counts[name] += 1

        # Sort the dictionary by message count
        sorted_message_counts = sorted(message_counts.items(), key=lambda item: item[1])

        N = 20  # Number of users with the most messages to display
        M = 5  # Number of users with the fewest messages to display

        # Extract the top N and bottom M users
        top_users = dict(list(sorted_message_counts)[-N:])
        bottom_users = dict(list(sorted_message_counts)[:M])

        # Merge the two dictionaries
        combined_users = {**bottom_users, **top_users}

        # Plotting the data
        plt.figure(figsize=(12, 8))
        names, counts = zip(*combined_users.items())
        plt.barh(names, counts, color='mediumseagreen')
        plt.xlabel('Number of Messages')
        plt.ylabel('Names')
        plt.title("Top and Bottom Users by Message Count")
        plt.gca().invert_yaxis()

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/whatsapp/message/activeusers/<int:num_users>', methods=['POST'])
def most_active_users(num_users):
    content, error_message, error_code = process_file(request)
    if content is None:
        return jsonify({'status': 'error', 'message': error_message}), error_code

    try:

        # Define a regex pattern to extract participant names from each message
        name_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?):")

        # Initialize a dictionary to count messages for each participant
        message_counts = defaultdict(int)

        # Iterate over each line to extract participant names and increment their message count
        for line in content:
            match = name_pattern.search(line)
            if match:
                name = match.group(1)
                message_counts[name] += 1

        # Sort participants based on message count and consider only top `num_users`
        sorted_counts = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:num_users]
        top_names = [item[0] for item in sorted_counts]
        top_message_counts = [item[1] for item in sorted_counts]

        # Create a bar chart to display the most active users
        plt.figure(figsize=(15, 8))
        sns.barplot(x=top_message_counts, y=top_names, palette="viridis")
        plt.title(f'Most Active {num_users} Users')
        plt.xlabel('Message Count')
        plt.ylabel('User')

        # Save the plot to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()

        # Send the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
