import emoji
import seaborn as sns
import numpy as np
import re
from collections import defaultdict
from collections import Counter
from flask import Flask, request, send_file, jsonify, abort
from wordcloud import WordCloud
from datetime import datetime
from LeIA import SentimentIntensityAnalyzer
from flask_cors import CORS
import tempfile
import io
import pandas as pd
import matplotlib
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

ALLOWED_HOSTS = ["https://analisegrupal.com.br", "https://api.analisegrupal.com.br"]

app = Flask(__name__)
CORS(app)

# s = SentimentIntensityAnalyzer()

# List of common Portuguese stop words
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
politics_keywords = {
    "presidente", "eleição", "governo", "voto", "partido", "política", 
    "senador", "deputado", "campanha", "ministro", "estado", "município", 
    "lei", "congresso", "constituição", "oposição", "reforma", "impeachment", 
    "legislação", "corrupção", "tribunal", "justiça", "candidato", "urna", "plebiscito"
}

religion_keywords = {
    "deus", "igreja", "rezar", "religião", "bíblia", "santo", "padre", 
    "oração", "fé", "espírito", "evangélico", "católico", "papa", 
    "culto", "missa", "milagre", "paróquia", "bispo", "pastor", "salmos", "crença"
}

soccer_keywords = {
    "golaco" ,"golaço", "gol", "jogo", "time", "placar", "futebol", "partida", "campeonato", 
    "jogador", "torcida", "estádio", "seleção", "treinador", "escalação", 
    "cartão", "falta", "pênalti", "liga", "derrota", "vitória", "empate", "goleiro", "taça"
}

sex_pornography_keywords = {
    "brotheragem", "holandês", "holandes", "puta", "gostosa", "safada", "linda", "pornografia", "nu", "nudez", "sensual", "erotismo", 
    "pornô", "fetiche", "prostituição", "adulto", "orgasmo", "lubrificante", 
    "preservativo", "camisinha", "vibrador", "strip", "lingerie", "sedução"
}

nltk.download('stopwords')
nltk.download('wordnet')

_stop_words = set(stopwords.words('portuguese'))
lemma = WordNetLemmatizer()

def preprocess(text):
    stop_words = _stop_words.union(portuguese_stop_words)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 3]
    tokens = [lemma.lemmatize(word) for word in tokens]
    return tokens

def build_lda_model(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary


@app.route('/')
def home():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden
    return jsonify({"message": "Hello, allowed host!"})


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify(success=True), 200

@app.route('/whatsapp/message/topic_modeling', methods=['POST'])
def topic_modeling():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    content = file.read().decode('utf-8').splitlines()

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

def determine_patterns(first_line):
    first_line = first_line.replace('"', "'")

    if "[" in first_line and "]" in first_line:
        date_pattern = r"\[.*?\]"
        message_pattern = r"(.*?):\s*(.*)"
    elif "," in first_line:
        date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}"
        message_pattern = r"- (.*?): (.*)"
    else:
        raise ValueError("Unsupported date format in the provided content.")

    return date_pattern, message_pattern

def extract_senders_messages_from_content(content):
    # Determine patterns based on the first line
    date_pattern, message_pattern = determine_patterns(content[0])

    # Debug info
    print("Date pattern:", date_pattern)
    print("Message pattern:", message_pattern)
    
    # Regular expression to extract timestamp, sender, and messages
    line_pattern = re.compile(rf"{date_pattern} {message_pattern}")
    
    extracted_data = []
    for line in content:
        if match := line_pattern.match(line):
            sender, message = match.group(1), match.group(2)
            extracted_data.append((sender, message))

    return extracted_data

def extract_timestamps_messages_from_content(content):
    # Replace double quotes with single quotes in the first line
    first_line = content[0].replace('"', "'")
    
    # Determine patterns based on the modified first line
    date_pattern, message_pattern = determine_patterns(first_line)

    # Debug info
    print("Date pattern:", date_pattern)
    print("Message pattern:", message_pattern)
    
    # Regular expression to extract timestamp, sender, and messages
    line_pattern = re.compile(rf"({date_pattern}) {message_pattern}")
    
    extracted_data = []
    for line in content:
        # Replace double quotes with single quotes for each line
        line = line.replace('"', "'")
        if match := line_pattern.match(line):
            timestamp, message = match.group(1), match.group(3)  # Adjusted the group numbers
            extracted_data.append((timestamp, message))

    return extracted_data


@app.route('/whatsapp/message/avg_sentiment_per_person', methods=['POST'])
def plot_avg_sentiment_per_person():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    # print(content[:5])

    extracted_data = extract_senders_messages_from_content(content)

    # print(extracted_data[:1])

    if len(extracted_data) == 0:  # Check if the DataFrame is empty
        print("NO DATA")
        return "No data available for plotting - avg_sentiment_per_person.empty", 400

    senders, messages = zip(*extracted_data)

    # Compute sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

    # Calculate average sentiment per person
    sender_sentiments = defaultdict(list)
    for sender, sentiment in zip(senders, sentiments):
        sender_sentiments[sender].append(sentiment)

    avg_sentiments = {sender: sum(vals)/len(vals) for sender, vals in sender_sentiments.items()}

    if len(avg_sentiments) == 0:  # Check if the DataFrame is empty
        return "No data available for plotting - avg_sentiments.empty", 400

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(avg_sentiments.keys(), avg_sentiments.values(), color='dodgerblue')
    plt.title("Average Sentiment Per Person")
    plt.ylabel("Average Sentiment Score")
    plt.xlabel("Sender")
    plt.xticks(rotation=45, ha='right')

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.tight_layout()
    plt.savefig(temp_file.name, format='png')
    plt.close()

    # Send the saved image file as the response
    return send_file(temp_file.name, mimetype='image/png')


@app.route('/whatsapp/message/length_over_time', methods=['POST'])
def plot_message_length_over_time():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    # print(content[:5])
    
    extracted_data = extract_timestamps_messages_from_content(content)

    # print(extracted_data[:1])

    if len(extracted_data) == 0:  # Check if the DataFrame is empty
        print("NO DATA")
        return "No data available for plotting - avg_sentiment_per_person.empty", 400

    timestamps, messages = zip(*extracted_data)

    # Compute message lengths
    message_lengths = [len(msg) for msg in messages]

    date_pattern, message_pattern = determine_patterns(content[0].replace('"', "'"))

    if date_pattern == r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}":
        datetime_format = '%m/%d/%y, %I:%M\u202f%p'
    elif date_pattern == r"\[.*?\]":
        datetime_format = '[%d/%m/%Y, %H:%M:%S]'
    else:
        datetime_format = None

    print(datetime_format)

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, errors='coerce', format=datetime_format),
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

@app.route('/whatsapp/message/avg_sentiment_per_person/json', methods=['POST'])
def avg_sentiment_per_person_json():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    extracted_data = extract_senders_messages_from_content(content)

    if len(extracted_data) == 0:  # Check if the DataFrame is empty
        print("NO DATA")
        return "No data available for plotting - avg_sentiment_per_person.empty", 400

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


@app.route('/whatsapp/message/sentiment_over_time', methods=['POST'])
def plot_sentiment_over_time():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    # print(content[:5])
    
    extracted_data = extract_timestamps_messages_from_content(content)

    # print(extracted_data[:1])

    if len(extracted_data) == 0:  # Check if the DataFrame is empty
        return "No data available for plotting", 400

    timestamps, messages = zip(*extracted_data)

    print(f"Timestamps count: {len(timestamps)}")

    # Compute sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(msg)['compound'] for msg in messages]

    print(f"Sentiments count: {len(sentiments)}")

    date_pattern, message_pattern = determine_patterns(content[0].replace('"', "'"))

    if date_pattern == r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}":
        datetime_format = '%m/%d/%y, %I:%M\u202f%p'
    elif date_pattern == r"\[.*?\]":
        datetime_format = '[%d/%m/%Y, %H:%M:%S]'
    else:
        datetime_format = None

    print(datetime_format)

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, errors='coerce', format=datetime_format),
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

@app.route('/whatsapp/message/peak_response_time', methods=['POST'])
def analyze_peak_response_time():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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

@app.route('/whatsapp/message/peak_response_time/json', methods=['POST'])
def analyze_peak_response_time_json():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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


@app.route('/whatsapp/message/activity_heatmap', methods=['POST'])
def activity_heatmap():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    date_pattern, message_pattern = determine_patterns(content[0].replace('"', "'"))

    if date_pattern == r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}":
        date_time_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2})\s[APMapm]{2}")
    elif date_pattern == r"\[.*?\]":
        date_time_pattern = re.compile(r"\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\]")
    else:
        return "No data available for plotting", 400

    # Extract date and time details
    date_times = []
    for line in content:
        match = date_time_pattern.search(line)
        if match:
            date_times.append(match.groups())

    # print(date_times[:0])
    # print(len(date_times))

    if len(date_times[0]) == 0:  # Check if the DataFrame is empty
        return "No data available for plotting", 400

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

@app.route('/whatsapp/message/user_activity_over_time', methods=['POST'])
def user_activity_over_time():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    date_pattern, message_pattern = determine_patterns(content[0].replace('"', "'"))

    # Adjust the regex pattern based on the determined patterns
    if date_pattern == r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}":
        activity_pattern = re.compile(rf"({date_pattern}) - (.*?): .*")
    elif date_pattern == r"\[.*?\]":
        activity_pattern = re.compile(rf"\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}:\d{1,2})\] (.*?): .*")
    else:
        return "No data available for plotting", 400

    # Extract dates and names
    dates_names = []
    for line in content:
        match = activity_pattern.search(line)
        if match:
            dates_names.append((match.group(1), match.group(2)))

    if len(dates_names) == 0:  # Check if the DataFrame is empty
        return "No data available for plotting", 400

    print(dates_names[:1])
    
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


@app.route('/whatsapp/message/top_emojis_json/<int:top_n>', methods=['POST'])
def get_top_emojis_json(top_n=10):
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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

# Conversational Turn Analysis
# In a group chat, it's interesting to see how often the conversation "turns" 
# to a new person. For example, if Person A sends 5 messages in a row, then Person B 
# sends 2 messages, and then Person C sends 1 message, there were 3 conversational turns. 
# This analysis can give insights into the flow of the conversation, indicating whether 
# it's dominated by long monologues or if it's more dynamic with many participants 
# chiming in frequently.
@app.route('/whatsapp/message/conversational_turns', methods=['POST'])
def plot_conversational_turns():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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

    # Prepare data for plotting
    sorted_turn_counts = sorted(turn_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_turn_counts)

    # Plotting the data
    plt.figure(figsize=(12, 8))
    plt.barh(names, counts, color='mediumseagreen')
    plt.xlabel('Number of Turns')
    plt.ylabel('Names')
    plt.title("Conversational Turns Analysis")
    plt.gca().invert_yaxis()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, format='png')
    plt.close()

    # Send the saved image file as the response
    return send_file(temp_file.name, mimetype='image/png')


# WEIRD
@app.route('/whatsapp/message/mention_analysis', methods=['POST'])
def mention_analysis():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()
    
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


@app.route('/whatsapp/message/active_days', methods=['POST'])
def plot_active_days():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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

@app.route('/whatsapp/message/topic_percentage', methods=['POST'])
def topic_percentage():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()
    
    # Extract messages from the content
    message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
    messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
    messages = [msg for msg in messages if msg is not None]

    # Check messages for presence of keywords
    topic_counts = {
        'politics': sum(1 for msg in messages if any(keyword in msg for keyword in politics_keywords)),
        'religion': sum(1 for msg in messages if any(keyword in msg for keyword in religion_keywords)),
        'soccer': sum(1 for msg in messages if any(keyword in msg for keyword in soccer_keywords)),
        'sex & pornography': sum(1 for msg in messages if any(keyword in msg for keyword in sex_pornography_keywords))
    }

    # Convert message counts to percentages
    total_messages = len(messages)
    topic_percentages = {topic: (count/total_messages)*100 for topic, count in topic_counts.items()}

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.bar(topic_percentages.keys(), topic_percentages.values(), color=['blue', 'green', 'red', 'purple'])
    plt.ylabel('Percentage of Messages (%)')
    plt.title('Percentage of Messages by Topic')
    
    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, format='png')
    plt.close()

    # Send the saved image file as the response
    return send_file(temp_file.name, mimetype='image/png')


@app.route('/whatsapp/message/topic_percentage/json', methods=['POST'])
def topic_percentage_json():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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


@app.route('/whatsapp/message/wordfrequency/<int:top_words>', methods=['POST'])
def plot_word_frequency(top_words=20):
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

    # Define a regex pattern to extract the content of each message
    message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")
    messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
    messages = [message for message in messages if message is not None]

    stop_words = _stop_words.union(portuguese_stop_words)

    # Tokenize the messages and count the frequency of each word
    word_freq = Counter()
    for message in messages:
        tokens = message.split()
        word_freq.update(tokens)

    # Remove stop words from the frequency counter
    for stop_word in stop_words:
        if stop_word in word_freq:
            del word_freq[stop_word]

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


@app.route('/whatsapp/message/wordcloud', methods=['POST'])
def plot_cleaned_wordcloud():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        # 1. Read the content
        content = file.read().decode('utf-8').splitlines()

        # Regular expression to extract the message content
        message_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - .*?: (.*)")

        # Extract message content
        messages = [message_pattern.search(line).group(1) if message_pattern.search(line) else None for line in content]
        messages = [message for message in messages if message is not None]

        stop_words = _stop_words.union(portuguese_stop_words)

        # Concatenate all messages
        text = ' '.join(messages)

        # Cleaning the text
        for stop_word in stop_words:
            text = text.replace(f" {stop_word} ", " ")
        text = text.replace("Media omitted", "").replace("omitted Media", "")

        # Generate the word cloud
        wordcloud = WordCloud(background_color='white', width=800, height=400, max_words=200).generate(text)

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

    return "Error processing the file", 500

@app.route('/whatsapp/message/lenghiest/top/<int:top_contributors>', methods=['POST'])
def plot_lengthiest_messages_pie_chart(top_contributors):
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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


@app.route('/whatsapp/message/sentiment/distribution', methods=['POST'])
def plot_sentiment_distribution():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the content of the uploaded file
    content = file.read().decode('utf-8').splitlines()

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


@app.route('/whatsapp/message/heatmap', methods=['POST'])
def plot_hourly_heatmap():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Read the file content
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        # 1. Read the content
        content = file.read().decode('utf-8').splitlines()

        # Extract hour along with the AM/PM marker using regex
        hour_ampm_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, (\d{1,2}:\d{1,2}\s[APMapm]{2}) - .*?:")
        hours_ampm = [hour_ampm_pattern.search(line).group(1) if hour_ampm_pattern.search(line) else None for line in content]
        hours_ampm = [hour for hour in hours_ampm if hour is not None]

        # Convert the extracted time to 24-hour format
        def convert_to_24_hour(time_str):
            return pd.to_datetime(time_str).hour

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

@app.route('/whatsapp/message/usercount', methods=['POST'])
def user_message_count():
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Initialize a dictionary to store the names and their message counts
    message_counts = defaultdict(int)

    # Check if a file was posted
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        # 1. Read the content
        content = file.read().decode('utf-8').splitlines()

        # Regular expression to extract the name pattern from a typical line
        name_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2} - (.*?):")

        # Extract names and count the messages
        for line in content:
            match = name_pattern.search(line)
            if match:
                name = match.group(1)
                message_counts[name] += 1

        # Convert the dictionary to a sorted list of tuples
        sorted_message_counts = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)

        # Filter out names with message counts below a certain threshold
        threshold = 100
        filtered_message_counts = [(name, count) for name, count in sorted_message_counts if count > threshold]

        # Data for the filtered pie chart
        filtered_names = [item[0] for item in filtered_message_counts]
        filtered_message_count_values = [item[1] for item in filtered_message_counts]

        # Plot the filtered pie chart
        plt.figure(figsize=(12, 8))
        plt.pie(filtered_message_count_values, labels=filtered_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title("Message Distribution by Users (Filtered)")
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # plt.show()

        # Convert the sorted_message_counts to a DataFrame for tabular representation
        df_message_counts = pd.DataFrame(sorted_message_counts, columns=["Name", "Message Count"])
        # print(df_message_counts)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name, format='png')
        plt.close()  # Close the plot

        # Return the saved image file as the response
        return send_file(temp_file.name, mimetype='image/png')

@app.route('/whatsapp/message/activeusers/<int:num_users>', methods=['POST'])
def most_active_users(num_users):
    # origin = request.headers.get('Origin') or request.headers.get('Referer')
    # if not origin or any(allowed_host in origin for allowed_host in ALLOWED_HOSTS):
    #     abort(403)  # Forbidden

    # Ensure a file is uploaded with the request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the uploaded file's content
    content = file.read().decode('utf-8').splitlines()

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
