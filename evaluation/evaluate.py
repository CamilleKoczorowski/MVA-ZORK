#!/usr/bin/env python3
"""
Evaluation Script for Text Adventure Agents

Evaluates student submissions by running their agent + MCP server
on a text adventure game for multiple trials and averaging scores.

Usage:
    # Evaluate a student submission
    python evaluation/evaluate.py \\
        --submission path/to/student/submission \\
        --game zork1 \\
        --trials 5 \\
        --max-steps 100

    # Evaluate with reference agent comparison
    python evaluation/evaluate.py \\
        --submission path/to/student/submission \\
        --game zork1 \\
        --reference

    # Evaluate from a Hugging Face Space
    python evaluation/evaluate.py \\
        --hf-space username/space-name \\
        --game zork1

    # Batch evaluate multiple submissions
    python evaluation/evaluate.py \\
        --submissions-dir path/to/all/submissions \\
        --game zork1 \\
        --output results.json

Examples:
    # Quick test with 3 trials
    python evaluation/evaluate.py -s ./submission_template -g zork1 -t 3

    # Full evaluation for grading
    python evaluation/evaluate.py -s ./submission_template -g advent -t 5 --max-steps 150
"""

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

# Suppress asyncio subprocess cleanup warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import EvaluationResult, TrialResult
from evaluation.runner import RunConfig, run_agent_with_server, run_reference_agent
from games.zork_env import list_available_games


def generate_seeds(base_seed: int, num_trials: int) -> list[int]:
    """Generate deterministic seeds for each trial."""
    random.seed(base_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(num_trials)]


async def evaluate_submission(
    submission_path: Path,
    game: str,
    num_trials: int = 5,
    max_steps: int = 100,
    base_seed: int = 42,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Evaluate a student submission across multiple trials.
    
    Args:
        submission_path: Path to student's submission directory
        game: Name of the game to evaluate on
        num_trials: Number of trials to run (default: 5)
        max_steps: Maximum steps per trial (default: 100)
        base_seed: Base seed for reproducibility (default: 42)
        verbose: Print detailed output
        
    Returns:
        EvaluationResult with aggregated metrics
    """
    # Locate agent and server files
    agent_path = submission_path / "agent.py"
    server_path = submission_path / "mcp_server.py"
    
    # Extract student ID from path or README
    student_id = submission_path.name
    readme_path = submission_path / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        # Try to extract student name from README
        for line in content.split("\n"):
            if line.startswith("# ") or "name:" in line.lower():
                student_id = line.replace("#", "").replace("name:", "").strip()[:50]
                break
    
    # Initialize results
    result = EvaluationResult(
        student_id=student_id,
        game=game,
        num_trials=num_trials,
        max_steps=max_steps,
    )
    
    # Generate deterministic seeds
    seeds = generate_seeds(base_seed, num_trials)
    
    print(f"\nEvaluating: {student_id}")
    print(f"Game: {game}")
    print(f"Trials: {num_trials}")
    print(f"Max steps: {max_steps}")
    print(f"Seeds: {seeds}")
    print("-" * 50)
    
    for i, seed in enumerate(seeds):
        trial_num = i + 1
        print(f"\nTrial {trial_num}/{num_trials} (seed={seed})...")
        
        config = RunConfig(
            agent_path=agent_path,
            server_path=server_path,
            game=game,
            max_steps=max_steps,
            seed=seed,
            verbose=verbose,
        )
        
        try:
            run_result = await run_agent_with_server(config)
            
            trial = TrialResult(
                trial_number=trial_num,
                final_score=run_result.final_score,
                max_score=run_result.max_score,
                moves=run_result.moves,
                locations_visited=len(run_result.locations_visited),
                game_completed=run_result.game_completed,
                error=run_result.error,
            )
            
            if run_result.error:
                print(f"  Error: {run_result.error[:100]}...")
            else:
                print(f"  Score: {run_result.final_score}")
                print(f"  Moves: {run_result.moves}")
                print(f"  Locations: {len(run_result.locations_visited)}")
                
        except Exception as e:
            trial = TrialResult(
                trial_number=trial_num,
                final_score=0,
                max_score=0,
                moves=0,
                locations_visited=0,
                game_completed=False,
                error=str(e),
            )
            print(f"  Exception: {e}")
        
        result.add_trial(trial)
    
    return result


async def evaluate_with_reference(
    submission_path: Path,
    game: str,
    num_trials: int = 5,
    max_steps: int = 100,
    base_seed: int = 42,
    verbose: bool = False,
) -> tuple[EvaluationResult, EvaluationResult]:
    """
    Evaluate student submission and compare with reference agent.
    
    Returns:
        Tuple of (student_result, reference_result)
    """
    # Evaluate student
    student_result = await evaluate_submission(
        submission_path=submission_path,
        game=game,
        num_trials=num_trials,
        max_steps=max_steps,
        base_seed=base_seed,
        verbose=verbose,
    )
    
    # Evaluate reference agent (from examples/mcp_react)
    print("\n" + "=" * 50)
    print("Running reference agent for comparison...")
    print("=" * 50)
    
    seeds = generate_seeds(base_seed, num_trials)
    
    reference_result = EvaluationResult(
        student_id="reference_agent",
        game=game,
        num_trials=num_trials,
        max_steps=max_steps,
    )
    
    for i, seed in enumerate(seeds):
        trial_num = i + 1
        print(f"\nReference Trial {trial_num}/{num_trials} (seed={seed})...")
        
        try:
            run_result = await run_reference_agent(
                game=game,
                max_steps=max_steps,
                seed=seed,
                verbose=verbose,
            )
            
            trial = TrialResult(
                trial_number=trial_num,
                final_score=run_result.final_score,
                max_score=run_result.max_score,
                moves=run_result.moves,
                locations_visited=len(run_result.locations_visited),
                game_completed=run_result.game_completed,
                error=run_result.error,
            )
            
            if run_result.error:
                print(f"  Error: {run_result.error[:100]}...")
            else:
                print(f"  Score: {run_result.final_score}")
                
        except Exception as e:
            trial = TrialResult(
                trial_number=trial_num,
                final_score=0,
                max_score=0,
                moves=0,
                locations_visited=0,
                game_completed=False,
                error=str(e),
            )
            print(f"  Exception: {e}")
        
        reference_result.add_trial(trial)
    
    return student_result, reference_result


def clone_hf_space(space_id: str, target_dir: Path) -> Path:
    """Clone a Hugging Face Space to local directory."""
    import subprocess
    
    # HF Spaces are git repos at huggingface.co/spaces/
    repo_url = f"https://huggingface.co/spaces/{space_id}"
    
    print(f"Cloning {repo_url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True,
        capture_output=True,
    )
    
    return target_dir


async def batch_evaluate(
    submissions_dir: Path,
    game: str,
    num_trials: int = 5,
    max_steps: int = 100,
    base_seed: int = 42,
    output_path: Path = None,
    verbose: bool = False,
) -> list[EvaluationResult]:
    """Evaluate all submissions in a directory."""
    results = []
    
    # Find all submission directories (those containing agent.py)
    submission_dirs = [
        d for d in submissions_dir.iterdir()
        if d.is_dir() and (d / "agent.py").exists()
    ]
    
    print(f"Found {len(submission_dirs)} submissions")
    
    for submission_path in sorted(submission_dirs):
        try:
            result = await evaluate_submission(
                submission_path=submission_path,
                game=game,
                num_trials=num_trials,
                max_steps=max_steps,
                base_seed=base_seed,
                verbose=verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to evaluate {submission_path}: {e}")
    
    # Sort by mean score (descending)
    results.sort(key=lambda r: r.mean_score, reverse=True)
    
    # Save results
    if output_path:
        output_data = {
            "evaluation_date": datetime.now().isoformat(),
            "game": game,
            "num_trials": num_trials,
            "max_steps": max_steps,
            "base_seed": base_seed,
            "results": [r.to_dict() for r in results],
            "leaderboard": [
                {
                    "rank": i + 1,
                    "student_id": r.student_id,
                    "mean_score": round(r.mean_score, 2),
                    "std_score": round(r.std_score, 2),
                }
                for i, r in enumerate(results)
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return results


def print_comparison(student: EvaluationResult, reference: EvaluationResult):
    """Print a comparison between student and reference results."""
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Student':<15} {'Reference':<15}")
    print("-" * 55)
    print(f"{'Mean Score':<25} {student.mean_score:<15.2f} {reference.mean_score:<15.2f}")
    print(f"{'Std Score':<25} {student.std_score:<15.2f} {reference.std_score:<15.2f}")
    print(f"{'Min Score':<25} {student.min_score:<15} {reference.min_score:<15}")
    print(f"{'Max Score':<25} {student.max_score_achieved:<15} {reference.max_score_achieved:<15}")
    print(f"{'Mean Moves':<25} {student.mean_moves:<15.1f} {reference.mean_moves:<15.1f}")
    print(f"{'Mean Locations':<25} {student.mean_locations:<15.1f} {reference.mean_locations:<15.1f}")
    print(f"{'Successful Trials':<25} {student.successful_trials:<15} {reference.successful_trials:<15}")
    
    # Performance ratio
    if reference.mean_score > 0:
        ratio = student.mean_score / reference.mean_score * 100
        print(f"\nStudent performance: {ratio:.1f}% of reference")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text adventure agent submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-s", "--submission",
        type=Path,
        help="Path to student submission directory",
    )
    input_group.add_argument(
        "--hf-space",
        type=str,
        help="Hugging Face Space ID (e.g., username/space-name)",
    )
    input_group.add_argument(
        "--submissions-dir",
        type=Path,
        help="Directory containing multiple submissions (for batch evaluation)",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "-g", "--game",
        type=str,
        default="lostpig",
        help="Game to evaluate on (default: lostpig)",
    )
    parser.add_argument(
        "-t", "--trials",
        type=int,
        default=5,
        help="Number of trials to run (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per trial (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    
    # Reference comparison
    parser.add_argument(
        "-r", "--reference",
        action="store_true",
        help="Also run reference agent (from examples/mcp_react) for comparison",
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List available games and exit",
    )
    
    args = parser.parse_args()
    
    # List games if requested
    if args.list_games:
        games = list_available_games()
        print(f"Available games ({len(games)}):")
        for game in games:
            print(f"  - {game}")
        return
    
    # Validate game
    available_games = list_available_games()
    if args.game not in available_games:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Available: {', '.join(available_games[:10])}...")
        sys.exit(1)
    
    # Handle HF Space input
    if args.hf_space:
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = clone_hf_space(args.hf_space, Path(tmpdir) / "submission")
            
            if args.reference:
                student_result, reference_result = asyncio.run(
                    evaluate_with_reference(
                        submission_path=submission_path,
                        game=args.game,
                        num_trials=args.trials,
                        max_steps=args.max_steps,
                        base_seed=args.seed,
                        verbose=args.verbose,
                    )
                )
                print_comparison(student_result, reference_result)
            else:
                result = asyncio.run(
                    evaluate_submission(
                        submission_path=submission_path,
                        game=args.game,
                        num_trials=args.trials,
                        max_steps=args.max_steps,
                        base_seed=args.seed,
                        verbose=args.verbose,
                    )
                )
                print("\n" + result.summary_str())
    
    # Handle batch evaluation
    elif args.submissions_dir:
        results = asyncio.run(
            batch_evaluate(
                submissions_dir=args.submissions_dir,
                game=args.game,
                num_trials=args.trials,
                max_steps=args.max_steps,
                base_seed=args.seed,
                output_path=args.output,
                verbose=args.verbose,
            )
        )
        
        # Print leaderboard
        print("\n" + "=" * 60)
        print("LEADERBOARD")
        print("=" * 60)
        print(f"\n{'Rank':<6} {'Student':<30} {'Mean Score':<12} {'Std':<10}")
        print("-" * 58)
        for i, r in enumerate(results):
            print(f"{i+1:<6} {r.student_id:<30} {r.mean_score:<12.2f} {r.std_score:<10.2f}")
    
    # Handle single submission
    else:
        submission_path = args.submission
        
        if not submission_path.exists():
            print(f"Error: Submission path not found: {submission_path}")
            sys.exit(1)
        
        if args.reference:
            student_result, reference_result = asyncio.run(
                evaluate_with_reference(
                    submission_path=submission_path,
                    game=args.game,
                    num_trials=args.trials,
                    max_steps=args.max_steps,
                    base_seed=args.seed,
                    verbose=args.verbose,
                )
            )
            print_comparison(student_result, reference_result)
            
            # Save results if output specified
            if args.output:
                output_data = {
                    "evaluation_date": datetime.now().isoformat(),
                    "student": student_result.to_dict(),
                    "reference": reference_result.to_dict(),
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")
        else:
            result = asyncio.run(
                evaluate_submission(
                    submission_path=submission_path,
                    game=args.game,
                    num_trials=args.trials,
                    max_steps=args.max_steps,
                    base_seed=args.seed,
                    verbose=args.verbose,
                )
            )
            
            print("\n" + result.summary_str())
            
            # Save results if output specified
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()