#!/usr/bin/env python3
"""
Convert R2E-Gym trajectory JSONL format to SWE-bench evaluation format.

Usage:
    python convert_r2egym_to_swebench.py <input_jsonl_file> [output_jsonl_file]
"""

import json
import sys
from pathlib import Path
from typing import Optional
from r2egym.agenthub.trajectory import Trajectory


def extract_model_name_from_filename(filename: str) -> str:
    """
    Extract model name from filename.
    
    Example: 'traj/r2egym-32b-agent-swebench-eval-20250602_124544.jsonl' -> 'r2egym-32b-agent'
    """
    path = Path(filename)
    name = path.stem  # Remove extension
    
    # Look for common patterns to extract model name
    # Pattern: {model_name}-{task}-{timestamp}
    parts = name.split('-')
    
    # Find where the task/eval part starts (like 'swebench', 'eval', etc.)
    task_indicators = ['swebench', 'eval', 'test', 'train']
    model_parts = []
    
    for part in parts:
        if any(indicator in part.lower() for indicator in task_indicators):
            break
        model_parts.append(part)
    
    if model_parts:
        return '-'.join(model_parts)
    
    # Fallback: use first part or default
    return parts[0] if parts else "R2E-Gym-Agent"


def convert_r2egym_to_swebench(
    input_file: str, 
    output_file: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Convert R2E-Gym trajectory JSONL to SWE-bench format.
    
    Args:
        input_file: Path to input JSONL file with R2E-Gym trajectories
        output_file: Path to output JSONL file (optional, auto-generated if None)
        model_name: Model name to use in SWE-bench format (optional, auto-detected if None)
        
    Returns:
        Path to the output file
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Extract model name from filename if not provided
    if model_name is None:
        model_name = extract_model_name_from_filename(input_file)
        print(f"Auto-detected model name: {model_name}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = str(input_path.with_suffix('.predictions.jsonl'))
    
    output_path = Path(output_file)
    
    print(f"Converting {input_file} to SWE-bench format...")
    print(f"Output will be saved to: {output_file}")
    
    converted_count = 0
    error_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Load trajectory from R2E-Gym format
                trajectory = Trajectory.load_from_model_dump_json(line)
                
                # Extract instance_id from docker_image or ds
                if trajectory.ds and 'instance_id' in trajectory.ds:
                    instance_id = trajectory.ds['instance_id']
                elif trajectory.ds and 'docker_image' in trajectory.ds:
                    # Extract instance_id from docker_image if not directly available
                    docker_image = trajectory.ds['docker_image']
                    # Try to extract instance_id from docker image name
                    # Format is typically: namanjain12/reponame:instance_id
                    if ':' in docker_image:
                        instance_id = docker_image.split(':')[-1]
                    else:
                        # Fallback: use docker_image as instance_id
                        instance_id = docker_image.replace('/', '_').replace(':', '_')
                else:
                    # Fallback: use docker_image from trajectory
                    instance_id = trajectory.docker_image.replace('/', '_').replace(':', '_')
                
                # Get the patch - prefer true_output_patch if available, otherwise output_patch
                try:
                    model_patch = trajectory.true_output_patch
                except:
                    model_patch = trajectory.output_patch
                
                # Create SWE-bench format entry
                swebench_entry = {
                    "model_name_or_path": model_name,
                    "instance_id": instance_id,
                    "model_patch": model_patch
                }
                
                # Write to output file
                outfile.write(json.dumps(swebench_entry) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
                continue
    
    print(f"Conversion completed!")
    print(f"Successfully converted: {converted_count} entries")
    print(f"Errors encountered: {error_count} entries")
    print(f"Output saved to: {output_file}")
    
    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_r2egym_to_swebench.py <input_jsonl_file> [output_jsonl_file] [model_name]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    model_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        output_path = convert_r2egym_to_swebench(input_file, output_file, model_name)
        print(f"\nSWE-bench format file ready for evaluation: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()