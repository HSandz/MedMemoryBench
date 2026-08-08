import argparse
import json
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    # randomly sample args.num indices from [args.start, args.end)
    indices = random.sample(range(args.start, args.end), args.num)
    indices = list(set(indices))
    indices.sort()
    print(indices)
    print("# of indices", len(indices))

    # save the indices to a file
    os.makedirs("outputs/longmemeval", exist_ok=True)
    output_path = f"outputs/longmemeval/longmemeval_s_dev_indices_{len(indices)}.json"
    with open(output_path, "w") as f:
        json.dump(indices, f)
    print("Indices saved to", output_path)
