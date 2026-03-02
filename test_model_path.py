import sys
import os
from pathlib import Path

# Mocking Flask for a simple path check
def test_path_resolution():
    try:
        # Replicate app.py logic
        BASE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = BASE_DIR / 'models' / 'perfect_gpu_model.pkl'
        
        print(f"DEBUG: BASE_DIR = {BASE_DIR}")
        print(f"DEBUG: MODEL_PATH = {MODEL_PATH}")
        
        if MODEL_PATH.exists():
            print("SUCCESS: Model file found at resolved path.")
            return True
        else:
            print("FAILURE: Model file NOT found at resolved path.")
            # List contents for debugging
            if (BASE_DIR / 'models').exists():
                 print(f"Contents of {BASE_DIR / 'models'}: {list((BASE_DIR / 'models').iterdir())}")
            else:
                 print(f"Models directory NOT found at {BASE_DIR / 'models'}")
            return False
    except Exception as e:
        print(f"ERROR during test: {e}")
        return False

if __name__ == "__main__":
    if test_path_resolution():
        sys.exit(0)
    else:
        sys.exit(1)
