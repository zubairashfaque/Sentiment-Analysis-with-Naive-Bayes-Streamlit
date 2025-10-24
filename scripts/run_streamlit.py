#!/usr/bin/env python3
"""
Script to run the Streamlit application.

Usage:
    python scripts/run_streamlit.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit application."""
    # Get the app path
    app_path = (
        Path(__file__).parent.parent
        / "src"
        / "sentiment_analysis"
        / "streamlit_app"
        / "app.py"
    )

    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    print("🚀 Starting Streamlit application...")
    print(f"📁 App path: {app_path}")
    print("🌐 The app will open in your browser automatically")
    print("⏹️  Press Ctrl+C to stop the server\n")

    # Run streamlit
    try:
        subprocess.run(
            ["streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "\n❌ Streamlit not found. Please install it:\n"
            "   pip install streamlit"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
