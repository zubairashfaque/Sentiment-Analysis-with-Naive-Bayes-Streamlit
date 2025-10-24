#!/usr/bin/env python3
"""
Script to run the FastAPI application.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8000 --reload
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Sentiment Analysis API"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    return parser.parse_args()


def main():
    """Run the FastAPI application."""
    args = parse_args()

    # Get the module path
    module_path = "sentiment_analysis.api.main:app"

    print("🚀 Starting Sentiment Analysis API...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {'enabled' if args.reload else 'disabled'}")
    print(f"👷 Workers: {args.workers}")
    print(f"\n🌐 API will be available at: http://{args.host}:{args.port}")
    print(f"📚 API docs at: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc at: http://{args.host}:{args.port}/redoc")
    print("\n⏹️  Press Ctrl+C to stop the server\n")

    # Build uvicorn command
    cmd = [
        "uvicorn",
        module_path,
        "--host", args.host,
        "--port", str(args.port),
        "--workers", str(args.workers),
    ]

    if args.reload:
        cmd.append("--reload")

    # Run uvicorn
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n✅ API stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running API: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "\n❌ uvicorn not found. Please install it:\n"
            "   pip install uvicorn[standard]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
