import sys
import os
sys.path.append(".")
from anki_web import get_collection

col = get_collection()
print("Buscando todos os cartões enterrados na coleção inteira...")
# queue: -2 = user buried, -3 = sched buried
buried_ids = col.db.list("select id from cards where queue in (-2, -3)")
print(f"Encontrados {len(buried_ids)} cartões enterrados!")

if buried_ids:
    # Restaura a fila do cartão para o tipo original
    # (0=new, 1=learn, 2=review, 3=relearn)
    # E atualiza a data de modificação (mod) e usn para sincronizar
    import time
    mod_time = int(time.time())
    col.db.execute(f"update cards set queue = type, mod = {mod_time}, usn = -1 where queue in (-2, -3)")
    
print("Pronto! Salvando alterações...")

tree = col.sched.deck_due_tree()
def print_tree(node, level=0):
    if node.new_count > 0 or node.learn_count > 0 or node.review_count > 0:
        print(f"{'  '*level}Name: {node.name}, new: {node.new_count}, learn: {node.learn_count}, review: {node.review_count}")
    for child in getattr(node, 'children', []):
        print_tree(child, level+1)

print_tree(tree)
