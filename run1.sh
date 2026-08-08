# source ./.venv/bin/activate

# python main.py -m persona_1/long_context_gemini -d medmemorybench --batch-api --batch-wait

# python main.py -m persona_1/bm25_rag_gemini -d medmemorybench --batch-api --batch-wait

python main.py -m persona_1/embedding_rag_gemini -d medmemorybench --batch-api --batch-wait

python main.py -m persona_1/graph_rag_gemini -d medmemorybench --batch-api --batch-wait