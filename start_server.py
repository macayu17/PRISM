"""
Start the Parkinson's Disease Assessment Web Application
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Parkinson's Disease Assessment Portal")
print("=" * 70)
print()

# Check dependencies
print("Checking dependencies...")
try:
    import flask
    print("[OK] Flask installed")
except ImportError:
    print("[ERROR] Flask not installed. Run: pip install flask")
    sys.exit(1)

try:
    import pandas
    print("[OK] Pandas installed")
except ImportError:
    print("[ERROR] Pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import numpy
    print("[OK] NumPy installed")
except ImportError:
    print("[ERROR] NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import sklearn
    print("[OK] Scikit-learn installed")
except ImportError:
    print("[ERROR] Scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

print()
print("All dependencies found!")
print()

# Change to src directory
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)  # Add src to path for imports

print("=" * 70)
print("Starting Web Server...")
print("=" * 70)
print()
print("The application will be available at:")
print("  → http://localhost:5000")
print()
print("Press Ctrl+C to stop the server")
print("=" * 70)
print()

# Import and run the app
from web_interface import app, initialize_system

# Check if user wants to skip initial model loading
import sys
skip_init = '--skip-init' in sys.argv or '-s' in sys.argv

if skip_init:
    print("[INFO] Skipping initial model loading (models will load on first prediction)")
    print("[INFO] This will make startup faster")
    print()
else:
    # Initialize the system
    print("Initializing AI models...")
    print("[INFO] This may take 1-2 minutes for the first time...")
    try:
        if initialize_system():
            print("[OK] System initialized successfully")
            print()
        else:
            print("[WARNING] System initialization had some issues")
            print("The app will still run and load models on first prediction")
            print()
    except KeyboardInterrupt:
        print("\n[INFO] Initialization interrupted by user")
        print("Starting server anyway (models will load on first prediction)")
        print()
    except Exception as e:
        print(f"[ERROR] Initialization error: {e}")
        print("Starting server anyway (models will load on first prediction)")
        print()

print("=" * 70)
print("🚀 SERVER STARTING...")
print("=" * 70)
print()

# Run the app
try:
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("Server stopped by user")
    print("=" * 70)
