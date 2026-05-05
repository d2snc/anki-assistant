import re
def strip_images_from_text(html_text):
    text = re.sub(r'<img[^>]*>', '', html_text, flags=re.IGNORECASE)
    text = re.sub(r'\[anki:play:[^\]]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[sound:[^\]]+\]', '', text, flags=re.IGNORECASE)
    return text

print(strip_images_from_text("Teste de pergunta [anki:play:q:0]"))
