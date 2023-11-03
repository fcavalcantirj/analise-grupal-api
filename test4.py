import re

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

def preprocess_content(content, words_to_remove=['Vic']):
    extracted_content = []

    # Define patterns to handle various date-time and message structures
    patterns = [
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?)\] (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}(:\d{1,2})?) ?(AM|PM)? - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2})\u202F(AM|PM) - (.*?): (.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}) \| (.*?) \| (.*)', re.IGNORECASE),
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?): \u200e?(.*)', re.IGNORECASE),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{1,2}) (\d{1,2}:\d{1,2}) - (.*?): \u200e?(.*)', re.IGNORECASE)
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

# Sample data lines for testing
sample_data = [
    "09/03/22 18:37 : Tiago Mendes Apu : Não dá pra colocar GNV?",
    "09/03/2022 18:37 | Tiago Mendes Apu | Não dá pra colocar GNV?",
    "09/03/2022 18:37 - Tiago Mendes Apu: Não dá pra colocar GNV?",
    "23/03/2018 00:31 - Vic Wolk: Tão no bar ainda??",
    "04/10/2023 06:40 - +55 61 9425-0939: 😭😭😭",
    "'02/10/2023 22:35 - As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo o WhatsApp pode ler ou ouvi-las. Toque para saber mais.'",
    "[09/05/2014, 23:50:59] 💞'Família do Ritiro' 💞: Some message",
    "02/10/2023 22:28:15 - +55 61 9335-9604: Another message",
    '1/5/21, 12:05 PM - Ricardo "Brave": Yet another message',
    "[21/04/2022 23:54:40] Cen COM MARIA MSM: ‎As mensagens e as chamadas são protegidas with the criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo the WhatsApp can read or hear them.",
    "[29/11/2022, 07:20:04] Dina: Bom dia!!",
    '5/25/21, 9:04 AM - Vic "Wolk" Zoop: caralho, maluco foi from sinner to saint dentro do mesmo minuto'
]

# Extracting the three parts: datetime, person, and message content and printing the results
extracted_data = preprocess_content(sample_data, [])
for item in extracted_data:
    print(item)
