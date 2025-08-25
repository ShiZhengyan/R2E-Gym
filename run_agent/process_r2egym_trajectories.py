#!/usr/bin/env python3
"""
Process R2E-Gym trajectory file into training data format.

This script:
1. Loads a trajectory file (JSONL format)
2. Filters for successful trajectories (reward == 1)
3. Formats them into conversation format for model training
4. Saves the results in HuggingFace dataset format
"""

import argparse
import os
import sys
from pathlib import Path
from datasets import Dataset
from typing import List, Dict, Any

from src.r2egym.agenthub.trajectory.trajectory import Trajectory
from src.r2egym.agenthub.agent.agent import AgentArgs


def load_trajectories_from_jsonl(jsonl_path: str) -> List[Trajectory]:
    """
    Load trajectories from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL trajectory file
        
    Returns:
        List of Trajectory objects
    """
    trajectories = []
    
    try:
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():  # Skip empty lines
                        traj = Trajectory.load_from_model_dump_json(line)
                        trajectories.append(traj)
                except Exception as e:
                    print(f"Error parsing line {line_num} in {jsonl_path}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_path}")
    except Exception as e:
        print(f"Error reading file {jsonl_path}: {e}")
        
    return trajectories


def filter_successful_trajectories(trajectories: List[Trajectory]) -> List[Trajectory]:
    """
    Filter trajectories to keep only successful ones (reward == 1).
    
    Args:
        trajectories: List of Trajectory objects
        
    Returns:
        List of successful trajectories
    """
    successful = []
    for traj in trajectories:
        if traj.reward == 1:
            successful.append(traj)
    
    print(f"Filtered {len(successful)} successful trajectories out of {len(trajectories)} total")
    return successful


def format_trajectory_for_training(traj: Trajectory, agent_args: AgentArgs) -> Dict[str, Any]:
    """
    Format a single trajectory into conversation format for model training.
    
    Args:
        traj: Trajectory object
        agent_args: AgentArgs object containing system and instance prompts
        
    Returns:
        Dictionary with conversation messages
    """
    # Build system prompt and user prompt using AgentArgs
    system_prompt = agent_args.system_prompt
    user_prompt = agent_args.instance_prompt.format(
        problem_statement=traj.problem_statement
    )
    
    # Initialize history as done in agent.py:815-818
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Process each trajectory step
    for step in traj.trajectory_steps:
        # Add assistant message with thought + action
        assistant_content = f"{step.thought}\n\n{step.action}"
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        # Add user message with observation (except for the last step if done=True)
        if not step.done:
            messages.append({
                "role": "user",
                "content": step.observation
            })
    
    return {
        "instance_id": traj.instance_name,
        "docker_image": traj.docker_image,
        "messages": messages,
        "reward": traj.reward,
        "num_steps": traj.num_steps,
        "total_tokens": traj.num_tokens_total,
        "agent_config": traj.agent_args.get('llm_name', 'unknown') if traj.agent_args else 'unknown'
    }


def process_traj_file(traj_file: str, output_path: str, config_path: str) -> None:
    """
    Process a single trajectory file.
    
    Args:
        traj_file: Path to trajectory JSONL file
        output_path: Path to save the processed dataset
        config_path: Path to config YAML file
    """
    traj_file = Path(traj_file)
    
    if not traj_file.exists():
        print(f"Error: Trajectory file does not exist: {traj_file}")
        return
    
    print(f"Processing trajectory file: {traj_file.name}")
    
    # Load agent configuration
    agent_args = AgentArgs.from_yaml(Path(config_path))
    print(f"Loaded agent config from: {config_path}")
    
    # Load trajectories from the file
    trajectories = load_trajectories_from_jsonl(str(traj_file))
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Filter successful trajectories
    successful_trajectories = filter_successful_trajectories(trajectories)
    
    print(f"\n=== Summary ===")
    print(f"Total trajectories loaded: {len(trajectories)}")
    print(f"Successful trajectories: {len(successful_trajectories)}")
    if len(trajectories) > 0:
        print(f"Success rate: {len(successful_trajectories) / len(trajectories) * 100:.2f}%")
    
    if not successful_trajectories:
        print("No successful trajectories found. Exiting.")
        return

    # Format trajectories for training
    print(f"\nFormatting {len(successful_trajectories)} successful trajectories...")
    formatted_data = []
    
    for traj in successful_trajectories:
        try:
            formatted = format_trajectory_for_training(traj, agent_args)
            if formatted is not None:
                formatted_data.append(formatted)
        except Exception as e:
            print(f"Error formatting trajectory {traj.instance_name}: {e}")
            continue
    
    print(f"Successfully formatted {len(formatted_data)} trajectories")
    
    if not formatted_data:
        print("No trajectories were successfully formatted. Exiting.")
        return
    
    # Create dataset
    dataset_dict = {
        'instance_id': [data['instance_id'] for data in formatted_data],
        'docker_image': [data['docker_image'] for data in formatted_data],
        'messages': [data['messages'] for data in formatted_data],
        'reward': [data['reward'] for data in formatted_data],
        'num_steps': [data['num_steps'] for data in formatted_data],
        'total_tokens': [data['total_tokens'] for data in formatted_data],
        'agent_config': [data['agent_config'] for data in formatted_data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    dataset.to_json(str(json_path))
    print(f"\nDataset saved to: {json_path}")
    
    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Number of examples: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Average steps per trajectory: {sum(dataset['num_steps']) / len(dataset):.2f}")
        print(f"Average tokens per trajectory: {sum(dataset['total_tokens']) / len(dataset):.2f}")
        
        # Agent config distribution
        agent_configs = {}
        for config in dataset['agent_config']:
            agent_configs[config] = agent_configs.get(config, 0) + 1
        print(f"Agent configurations: {dict(agent_configs)}")
    
    # Show sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== Sample Example ===")
        print(f"Instance ID: {sample['instance_id']}")
        print(f"Docker Image: {sample['docker_image']}")
        print(f"Agent Config: {sample['agent_config']}")
        print(f"Number of messages: {len(sample['messages'])}")
        print(f"First few messages:")
        for i, msg in enumerate(sample['messages'][:3]):
            content_preview = msg['content'][:200].replace('\n', '\\n')
            print(f"  {i+1}. {msg['role']}: {content_preview}...")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process R2E-Gym trajectory files into training data format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With explicit output path
  python process_r2egym_trajectories.py \\
    --traj-file ./traj/run-debug.jsonl \\
    --output ./data/r2egym_training_data.json \\
    --config-path ./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml
  
  # Auto-generate output path (will create ./traj/processed/run-debug_training_data.json)
  python process_r2egym_trajectories.py --traj-file ./traj/run-debug.jsonl
        """
    )
    
    parser.add_argument(
        '--traj-file',
        required=True,
        help='Path to trajectory JSONL file'
    )
    
    parser.add_argument(
        '--output',
        help='Output path for the processed dataset (JSON format). If not specified, will auto-generate based on input file name.'
    )
    
    parser.add_argument(
        '--config-path',
        default='src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml',
        help='Path to config YAML file (default: src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.traj_file):
        print(f"Error: Trajectory file does not exist: {args.traj_file}")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file does not exist: {args.config_path}")
        sys.exit(1)
    
    # Generate output path if not specified
    output_path = args.output
    if not output_path:
        input_path = Path(args.traj_file)
        output_name = f"{input_path.stem}_training_data.json"
        output_path = input_path.parent / "processed" / output_name
        print(f"No output path specified, using: {output_path}")
    
    # Process trajectories
    process_traj_file(args.traj_file, output_path, args.config_path)


if __name__ == "__main__":
    main()