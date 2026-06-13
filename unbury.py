import sys
sys.path.append(".")
from anki_web import get_collection

col = get_collection()
print("Desenterrando cartões de todo o deck...")

# Em schedV2 / V3:
if hasattr(col.sched, 'unburyCardsForDeck'):
    col.sched.unburyCardsForDeck()
elif hasattr(col.sched, 'unbury_deck'):
    col.sched.unbury_deck()
else:
    # Achar cartões enterrados:
    buried_ids = col.db.list("select id from cards where queue = -2")
    if buried_ids:
        col.sched.unbury_cards(buried_ids)

col.save()
print("Pronto! Cartões desenterrados.")

tree = col.sched.deck_due_tree()
def print_tree(node, level=0):
    print(f"{'  '*level}Name: {node.name}, new: {node.new_count}, learn: {node.learn_count}, review: {node.review_count}")
    for child in getattr(node, 'children', []):
        print_tree(child, level+1)

print_tree(tree)
