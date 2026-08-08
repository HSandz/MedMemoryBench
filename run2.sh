# source ./.venv/bin/activate

python main.py -m persona_1/amem_gemini -d medmemorybench --batch-api --batch-wait --resume

python main.py -m persona_1/mem0_gemini -d medmemorybench --batch-api --batch-wait

python main.py -m persona_1/memos_gemini -d medmemorybench --batch-api --batch-wait

python main.py -m persona_1/lightmem_gemini -d medmemorybench --batch-api --batch-wait
