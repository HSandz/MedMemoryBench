#!/usr/bin/env python3
"""
Script to load the ToT (Test of Time) dataset from Hugging Face,
extract the tot_semantic subset, and randomly sample 100 examples.

Dataset: https://huggingface.co/datasets/baharef/ToT
"""

import random
import json
from datasets import load_dataset
import pandas as pd


def load_tot_semantic_dataset(num_samples=100, random_seed=42):
    """
    Load the ToT dataset, extract tot_semantic subset, and randomly sample examples.
    
    Args:
        num_samples (int): Number of samples to randomly select (default: 100)
        random_seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
        list: List of sampled examples from tot_semantic dataset
    """
    print("Loading ToT dataset from Hugging Face...")
    
    try:
        # Load the tot_semantic subset directly
        dataset = load_dataset("baharef/ToT", "tot_semantic")
        
        # The dataset has different splits, let's check what's available
        print(f"Available splits: {list(dataset.keys())}")
        
        # Get the test split (most common split for evaluation datasets)
        if "test" in dataset:
            tot_semantic = dataset["test"]
        elif "train" in dataset:
            tot_semantic = dataset["train"]
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            tot_semantic = dataset[split_name]
            print(f"Using split: {split_name}")
        
        print(f"Total examples in tot_semantic: {len(tot_semantic)}")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Convert to list for easier sampling
        data_list = list(tot_semantic)
        
        # Randomly sample the specified number of examples
        sampled_data = random.sample(data_list, min(num_samples, len(data_list)))
        
        print(f"Successfully sampled {len(sampled_data)} examples from tot_semantic dataset")
        
        return sampled_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def save_sampled_dataset(sampled_data, output_file="tot_semantic_sampled.json"):
    """
    Save the sampled dataset to a JSON file.
    
    Args:
        sampled_data (list): List of sampled examples
        output_file (str): Output file path
    """
    if sampled_data is None:
        print("No data to save.")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2, ensure_ascii=False)
        print(f"Sampled dataset saved to {output_file}")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def analyze_dataset_structure(sampled_data):
    """
    Analyze and display the structure of the sampled dataset.
    
    Args:
        sampled_data (list): List of sampled examples
    """
    if not sampled_data:
        print("No data to analyze.")
        return
    
    print("\n=== Dataset Analysis ===")
    print(f"Number of samples: {len(sampled_data)}")
    
    # Check fields in the first example
    if sampled_data:
        first_example = sampled_data[0]
        print(f"Fields in each example: {list(first_example.keys())}")
        
        # Display first few examples
        print("\n=== Sample Examples ===")
        for i, example in enumerate(sampled_data[:3]):
            print(f"\nExample {i+1}:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
    
    # Check question types if available
    if sampled_data and 'question_type' in sampled_data[0]:
        question_types = {}
        for example in sampled_data:
            q_type = example.get('question_type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        print(f"\n=== Question Types Distribution ===")
        for q_type, count in question_types.items():
            print(f"  {q_type}: {count}")


def main():
    """Main function to execute the dataset loading and sampling."""
    print("=== ToT Semantic Dataset Loader ===")
    
    # Load and sample the dataset
    sampled_data = load_tot_semantic_dataset(num_samples=100, random_seed=42)
    
    if sampled_data:
        # Analyze the dataset structure
        analyze_dataset_structure(sampled_data)
        
        # Save the sampled dataset
        save_sampled_dataset(sampled_data, "tot_semantic_sampled_100.json")
        
        print("\n=== Summary ===")
        print(f"✓ Successfully loaded and sampled 100 examples from tot_semantic dataset")
        print(f"✓ Data saved to 'tot_semantic_sampled_100.json'")
        print(f"✓ Dataset ready for use!")
    else:
        print("❌ Failed to load the dataset. Please check the error messages above.")


if __name__ == "__main__":
    main()