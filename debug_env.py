import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Transformers File: {transformers.__file__}")
    print("Attempting to import AutoTokenizer...")
    from transformers import AutoTokenizer
    print("SUCCESS: AutoTokenizer imported!")
except ImportError as e:
    print(f"ERROR: {e}")
    if 'transformers' in locals():
        print(f"Transformers dir contents: {dir(transformers)}")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

try:
    import streamlit
    print(f"Streamlit Version: {streamlit.__version__}")
except ImportError:
    print("Streamlit not found.")
