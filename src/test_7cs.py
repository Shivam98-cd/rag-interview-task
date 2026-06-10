import sys
sys.path.insert(0, r"D:\rag-interview-task")

from src.retrieval.retriever import retrieve

results = retrieve("What are the 7 Cs of effective communication?")
print(f"\nTop {len(results)} chunks retrieved:\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] score={r['score']} | page={r['page']}")
    print(f"     {r['text'][:150]}")
    print()