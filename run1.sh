# source ./.venv/bin/activate

# python main.py -m long_context_gemini -d medmemorybench_1 --batch-api

python main.py -m bm25_rag_gemini -d medmemorybench_1 --batch-api

python main.py -m embedding_rag_gemini -d medmemorybench_1 --batch-api

# python main.py -m graph_rag_gemini -d medmemorybench_1 --batch-api