#!/usr/bin/env python3
"""Test runner script for the Slack Knowledge Bot testing framework."""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, cwd: str = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    return result.returncode


def setup_test_environment():
    """Set up the test environment."""
    print("🔧 Setting up test environment...")
    
    # Set test environment variables
    os.environ["TEST_MODE"] = "true"
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:admin@localhost:5433/slackbot_test"
    os.environ["REDIS_URL"] = "redis://localhost:6380/15"
    os.environ["OPENAI_API_KEY"] = "test_key_12345"
    os.environ["SLACK_BOT_TOKEN"] = "xoxb-test-token"
    
    print("✅ Test environment configured")


def run_docker_tests():
    """Run tests in Docker environment."""
    print("🐳 Running tests in Docker environment...")
    
    # Build and run tests
    cmd = "docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit"
    return run_command(cmd)


def run_local_tests(test_type: str = "all", verbose: bool = False):
    """Run tests locally."""
    print(f"🧪 Running {test_type} tests locally...")
    
    # Install test dependencies
    print("📦 Installing test dependencies...")
    run_command("pip install pytest pytest-asyncio pytest-cov")
    
    # Build test command
    cmd = "pytest tests/"
    
    if test_type == "unit":
        cmd += " -m unit"
    elif test_type == "integration":
        cmd += " -m integration"
    elif test_type == "e2e":
        cmd += " -m e2e"
    elif test_type == "quality":
        cmd += " -m quality"
    
    if verbose:
        cmd += " -v"
    
    cmd += " --tb=short"
    
    return run_command(cmd)


def run_specific_tests(test_path: str, verbose: bool = False):
    """Run specific test file or directory."""
    print(f"🎯 Running specific tests: {test_path}")
    
    cmd = f"pytest {test_path}"
    if verbose:
        cmd += " -v"
    
    cmd += " --tb=short"
    
    return run_command(cmd)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Slack Knowledge Bot Test Runner")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "e2e", "quality", "docker", "specific"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--path",
        help="Specific test path (for 'specific' test type)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up test environment without running tests"
    )
    
    args = parser.parse_args()
    
    # Set up test environment
    setup_test_environment()
    
    if args.setup_only:
        print("✅ Test environment setup complete")
        return 0
    
    # Run tests based on type
    if args.test_type == "docker":
        exit_code = run_docker_tests()
    elif args.test_type == "specific":
        if not args.path:
            print("❌ Error: --path is required for specific test type")
            return 1
        exit_code = run_specific_tests(args.path, args.verbose)
    else:
        exit_code = run_local_tests(args.test_type, args.verbose)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
