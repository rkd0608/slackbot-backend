#!/usr/bin/env python3
"""Run initial database migration."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
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
    """Main migration function."""
    print("🚀 Running initial database migration...")
    
    # Check if we're in the right directory
    if not Path("alembic").exists():
        print("❌ alembic directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if database is running
    print("📊 Checking database connection...")
    if not run_command("docker-compose ps db", "Checking database status"):
        print("❌ Database is not running. Please start it first:")
        print("   docker-compose up -d db")
        sys.exit(1)
    
    # Create initial migration
    print("\n📝 Creating initial migration...")
    if not run_command("alembic revision --autogenerate -m 'init schema'", "Creating migration"):
        print("⚠️  Migration creation failed. This might be normal if no changes detected.")
    
    # Run the migration
    print("\n🚀 Running migration...")
    if not run_command("alembic upgrade head", "Running migration"):
        print("❌ Migration failed. Check your database connection and configuration.")
        sys.exit(1)
    
    print("\n🎉 Initial migration completed successfully!")
    print("📊 Database schema is now up to date.")
    print("💡 To check migration status: alembic current")
    print("💡 To create new migrations: alembic revision --autogenerate -m 'description'")


if __name__ == "__main__":
    main()
