import sys
sys.path.append(".")
from anki_web import get_collection
col = get_collection()
tree = col.sched.deck_due_tree()
def print_tree(node, level=0):
    print(f"{'  '*level}Name: {node.name}, new: {node.new_count}, learn: {node.learn_count}, review: {node.review_count}")
    for child in node.children:
        print_tree(child, level+1)
print_tree(tree)
