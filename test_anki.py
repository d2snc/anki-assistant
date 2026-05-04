from anki.collection import Collection
from config import ANKI_PATH
col = Collection(ANKI_PATH)
print("select" in dir(col.decks))
print("set_current" in dir(col.decks))
