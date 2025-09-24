#!/usr/bin/env python3
"""Setup script for pre-commit hooks."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up pre-commit hooks...")
    
    # Check if we're in the right directory
    if not Path(".pre-commit-config.yaml").exists():
        print("❌ .pre-commit-config.yaml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install pre-commit
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        sys.exit(1)
    
    # Install the git hook scripts
    if not run_command("pre-commit install", "Installing git hooks"):
        sys.exit(1)
    
    # Run pre-commit on all files
    if not run_command("pre-commit run --all-files", "Running pre-commit on all files"):
        print("⚠️  Some pre-commit checks failed. This is normal for the first run.")
        print("   The hooks are now installed and will run on future commits.")
    
    print("\n🎉 Pre-commit setup complete!")
    print("📝 Hooks will now run automatically on every commit.")
    print("💡 To run manually: pre-commit run --all-files")


if __name__ == "__main__":
    main()
