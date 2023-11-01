import re

content_line = "[09/05/2014, 23:50:59] 💞'Família do Ritiro' 💞: ‎As mensagens e as chamadas são protegidas com a criptografia de ponta a ponta e ficam somente entre você e os participantes desta conversa. Nem mesmo o WhatsApp pode ler ou ouvi-las."
message_pattern = r"- (.*?): (.*)"
match = re.search(message_pattern, content_line)

if match:
    sender = match.group(1)
    message = match.group(2)
    print(f"Sender: {sender}")
    print(f"Message: {message}")
else:
    print("Pattern did not match.")
