"""
NexusAI Unified Launcher Script.
Supports launching the FastAPI backend, Streamlit frontend, or both concurrently.
Usage:
    python run.py           # Launches both backend and frontend
    python run.py backend   # Launches only FastAPI backend
    python run.py frontend  # Launches only Streamlit frontend
    python run.py test      # Runs automated Pytest suite
"""
import os
from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_backend():
    """Runs the FastAPI backend via uvicorn."""
    print("🚀 Starting NexusAI FastAPI Backend on http://127.0.0.1:8000 ...")
    cmd = [
        PYTHON_EXE,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    return subprocess.Popen(cmd, cwd=str(BASE_DIR))


def run_frontend():
    """Runs the Streamlit frontend."""
    print("✨ Starting NexusAI Streamlit Frontend on http://localhost:8501 ...")
    cmd = [
        PYTHON_EXE,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "frontend" / "app.py"),
        "--server.port=8501",
        "--server.address=127.0.0.1",
    ]
    return subprocess.Popen(cmd, cwd=str(BASE_DIR))


def run_tests():
    """Runs pytest test suite."""
    print("🧪 Running NexusAI Automated Test Suite...")
    cmd = [PYTHON_EXE, "-m", "pytest", "tests/", "-v", "--tb=short"]
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    sys.exit(result.returncode)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode == "backend":
        proc = run_backend()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()

    elif mode == "frontend":
        proc = run_frontend()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()

    elif mode == "test":
        run_tests()

    elif mode in ("all", "start"):
        backend_proc = run_backend()
        time.sleep(2)  # Give backend a moment to start
        frontend_proc = run_frontend()

        print("\n" + "=" * 65)
        print("  NexusAI – Agentic RAG Research Assistant is Running!")
        print("  Backend API:   http://127.0.0.1:8000")
        print("  API Docs:      http://127.0.0.1:8000/docs")
        print("  Frontend UI:   http://localhost:8501")
        print("=" * 65 + "\n")
        print("Press Ctrl+C to stop all services.\n")

        try:
            while True:
                time.sleep(1)
                if backend_proc.poll() is not None:
                    print("Backend process terminated unexpectedly.")
                    break
                if frontend_proc.poll() is not None:
                    print("Frontend process terminated unexpectedly.")
                    break
        except KeyboardInterrupt:
            print("\nShutting down NexusAI services...")
        finally:
            backend_proc.terminate()
            frontend_proc.terminate()
            backend_proc.wait()
            frontend_proc.wait()
            print("Services stopped.")

    else:
        print(f"Unknown command: '{mode}'. Usage: python run.py [all|backend|frontend|test]")
        sys.exit(1)


if __name__ == "__main__":
    main()
