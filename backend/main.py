"""
Main entry point for Data Synthesizer
Supports both CLI and API modes
"""

import sys
import os
import argparse

# UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="AI Data Synthesizer - Generate Synthetic Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Start CLI mode
  python main.py --mode cli
  python main.py --mode api
  python main.py --mode api --port 8080
        """
    )

    parser.add_argument(
        "--mode",
        choices=["cli", "api"],
        default="cli",
        help="Mode to run the application"
    )

    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "api":
        run_api(args.host, args.port, args.reload)


def run_cli():
    try:
        from cli import main as cli_main
        cli_main()
    except Exception as e:
        print(f"CLI Error: {e}")
        sys.exit(1)


def run_api(host, port, reload):
    try:
        import uvicorn
        print("Starting API server...")
        uvicorn.run(
            "api.fast_api:app",
            host=host,
            port=port,
            reload=reload,
        )
    except Exception as e:
        print(f"API Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()