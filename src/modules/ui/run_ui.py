#!/usr/bin/env python3
"""
Quick launcher for the Streamlit UI
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    
    print("🎬 Starting Content-Pal Streamlit UI...")
    print("📍 Make sure your Docker services are running:")
    print("   docker-compose up -d")
    print()
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "streamlit_app.py")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "requests", "pandas"])
    
    # Launch streamlit
    try:
        subprocess.run([
            "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print(f"💡 Try running directly: streamlit run {app_path}")

if __name__ == "__main__":
    main()