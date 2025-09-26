#!/usr/bin/env python3
"""Database setup and migration script."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up database...")
    
    # Check if we're in the right directory
    if not Path("alembic").exists():
        print("❌ alembic directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if alembic.ini exists
    if not Path("alembic.ini").exists():
        print("❌ alembic.ini not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install alembic if not already installed
    if not run_command("pip install alembic", "Installing alembic"):
        sys.exit(1)
    
    # Initialize alembic (if not already initialized)
    if not Path("alembic/versions").exists():
        if not run_command("alembic init alembic", "Initializing alembic"):
            print("⚠️  Alembic already initialized or failed to initialize.")
    
    # Run database migrations
    print("\n📊 Running database migrations...")
    if not run_command("alembic upgrade head", "Running migrations"):
        print("⚠️  Migrations failed. Check your database connection and configuration.")
        sys.exit(1)
    
    print("\n🎉 Database setup complete!")
    print("📝 Database tables created successfully.")
    print("💡 To run migrations manually: alembic upgrade head")
    print("💡 To create new migration: alembic revision --autogenerate -m 'description'")


if __name__ == "__main__":
    main()
