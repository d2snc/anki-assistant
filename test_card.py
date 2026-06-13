import sys
sys.path.append(".")
from anki_web import get_collection
import traceback

col = get_collection()
card = col.sched.getCard()
if card:
    try:
        print("Fields:", card.note().fields)
        print("Question():", card.question())
        print("Answer():", card.answer())
    except Exception as e:
        print("Error getting question/answer:")
        traceback.print_exc()
else:
    print("No cards due")
