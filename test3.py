# import re

# def determine_patterns(first_line):
#     if "[" in first_line and "]" in first_line:
#         date_pattern = r"\[.*?\]"
#     elif "," in first_line:
#         date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{1,2}\s[APMapm]{2}"
#     elif " " in first_line:
#         date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{1,2}:\d{1,2}"
#     else:
#         raise ValueError("Unsupported date format in the provided content.")
    
#     message_pattern = r"- (.*?): (.*)"
#     return date_pattern, message_pattern

# def preprocess_content(content):
#     processed_content = []
#     for line in content:
#         # Replace double quotes with single quotes
#         line = line.replace('"', "'")
        
#         date_pattern, _ = determine_patterns(line)

#         # Pattern to match the format like: [09/05/2014, 23:50:59]
#         if date_pattern == r"\[.*?\]":
#             pattern1 = re.compile(r"\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?):")
#             line = pattern1.sub(r"\1, \2 - \3:", line)

#             # Pattern to match the format like: [21/04/2022 23:54:40]
#             pattern3 = re.compile(r"\[(\d{1,2}/\d{1,2}/\d{2,4}) (\d{1,2}:\d{1,2}:\d{1,2})\] (.*?):")
#             line = pattern3.sub(lambda m: f"{m.group(1)}, {m.group(2)[:-3]} - {m.group(3)}:", line)

#         # Add the processed line to the output list
#         processed_content.append(line)
        
#     return processed_content

# # Sample data lines for testing
# sample_data = [
#     " ",
#     "'02/10/2023 22:35 - As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo o WhatsApp pode ler ou ouvi-las. Toque para saber mais.'",
#     "[09/05/2014, 23:50:59] 💞'Família do Ritiro' 💞: Some message",
#     "02/10/2023 22:28:15 - +55 61 9335-9604: Another message",
#     '1/5/21, 12:05 PM - Ricardo "Brave": Yet another message',
#     "[21/04/2022 23:54:40] Cen COM MARIA MSM: ‎As mensagens e as chamadas são protegidas with the criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo the WhatsApp can read or hear them.",
#     "[29/11/2022, 07:20:04] Dina: Bom dia!!",
#     '5/25/21, 9:04 AM - Vic "Wolk" Zoop: caralho, maluco foi from sinner to saint dentro do mesmo minuto'
# ]

# # Applying the preprocessing function and printing the results
# processed_data = preprocess_content(sample_data)
# for line in processed_data:
#     print(line)
import re

def preprocess_content(content):
    processed_content = []

    for line in content:
        # Replace double quotes with single quotes
        line = line.replace('"', "'")

        # Use regex to extract date, time, person, and content
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}),? (\d{1,2}:\d{1,2}(:\d{1,2})?) ?(AM|PM)? - (.*?): (.*)', line)
        
        if not match:
            continue

        date, time, _, ampm, person, content = match.groups()
        
        # Convert time to 12-hour format with AM/PM
        hour, minute = map(int, time.split(':')[:2])
        if not ampm:
            ampm = "AM" if hour < 12 else "PM"
            if hour > 12:
                hour -= 12
        formatted_time = f"{hour:02}:{minute:02} {ampm}"

        # Ensure the date is in 'dd/mm/yy' format
        day, month, year = map(int, date.split('/'))
        formatted_date = f"{day:02}/{month:02}/{year%100:02}"

        # Format the processed line
        processed_line = f"'{formatted_date}, {formatted_time} - {person}: {content}'"
        processed_content.append(processed_line)

    return processed_content

# Sample data lines for testing
sample_data = [
    "'02/10/2023 22:35 - As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo o WhatsApp pode ler ou ouvi-las. Toque para saber mais.'",
    "[09/05/2014, 23:50:59] 💞'Família do Ritiro' 💞: Some message",
    "02/10/2023 22:28:15 - +55 61 9335-9604: Another message",
    '1/5/21, 12:05 PM - Ricardo "Brave": Yet another message',
    "[21/04/2022 23:54:40] Cen COM MARIA MSM: ‎As mensagens e as chamadas são protegidas with the criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo the WhatsApp can read or hear them.",
    "[29/11/2022, 07:20:04] Dina: Bom dia!!",
    '5/25/21, 9:04 AM - Vic "Wolk" Zoop: caralho, maluco foi from sinner to saint dentro do mesmo minuto'
]

# Applying the preprocessing function and printing the results
processed_data = preprocess_content(sample_data)
for line in processed_data:
    print(line)


