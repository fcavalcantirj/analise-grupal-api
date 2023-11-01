import re

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

# Test the function with a sample content
content = [
    "[09/05/2014, 23:50:59] 💞'Família do Ritiro' 💞: ‎As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo o WhatsApp pode ler ou ouvi-las."
]

print("First line:", content[0])
print(extract_timestamps_messages_from_content(content))
print(extract_senders_messages_from_content(content))
