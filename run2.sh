# source ./.venv/bin/activate

python main.py --stage query --memory-run 20260909_101050 -m test/event_state_gemini -d locomo

python main.py --stage query --memory-run 20260909_101050 -m test/event_state_gemini1 -d locomo --batch-api

python main.py --stage query --memory-run 20260909_101050 -m test/event_state_gemini2 -d locomo --batch-api

python main.py --stage query --memory-run 20260909_101050 -m test/event_state_gemini3 -d locomo --batch-api

python main.py --stage query --memory-run 20260909_101050 -m test/event_state_gemini4 -d locomo --batch-api