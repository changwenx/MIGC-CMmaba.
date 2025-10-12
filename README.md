🧠 #MIGC-CMmaba.
# 🚦 Traffic Flow Prediction Based on Mamba Architecture

> A traffic flow prediction project leveraging **Mamba architecture**, supporting **multi-scale time series image processing** and **spatio-temporal feature fusion** for accurate and efficient traffic forecasting.

---

## 📁 Project Structure

```text
.
├── mainjinan.py / mainla.py / main03.py   # Main entry points for different datasets
├── data_preprocessing.py                   # Data preprocessing module
├── multiscale_processor.py                 # Multi-scale time series processing
├── spatio.py                               # Spatial feature extraction
├── st_fusion.py                            # Spatio-temporal fusion model
├── mamba_vision_model.py                   # Mamba-based vision model
├── trainer.py                              # Training pipeline
├── check_environment.py                    # Environment validation script
└── results/                                # Output directory
    ├── PEMS03/                             # PEMS03 dataset results
    └── experiment_summary.txt              # Summary of experiments
```


## ⚙️ Environment Configuration

| Dependency | Version |
|-------------|----------|
| Python | 3.8.20 |
| PyTorch | 2.2.2 |
| mamba-ssm | 1.1.3 |
| causal-conv1d | 1.1.3 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |

---

## 🧩 Environment Validation

Before running the project, please verify your environment.

### 1️⃣ Create `check_environment.py`


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Validation Script for Traffic Flow Prediction Project
Based on Mamba Architecture
---------------------------------------------------------------
Verifies that all dependencies are correctly installed and accessible.
"""

import sys
import torch
import numpy as np
import pandas as pd

def check_environment():
    print("🔍 Checking environment configuration...\n")

    try:
        import mamba_ssm
        import causal_conv1d

        print("✅ All dependencies are installed correctly!\n")

        print(f"🐍 Python version: {sys.version.split()[0]}")
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🧩 mamba-ssm version: {getattr(mamba_ssm, '__version__', 'unknown')}")
        print(f"🔄 causal-conv1d version: {getattr(causal_conv1d, '__version__', 'unknown')}")
        print(f"🔢 numpy version: {np.__version__}")
        print(f"📊 pandas version: {pd.__version__}")

        # CUDA info
        print("\n💻 CUDA information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            device_id = torch.cuda.current_device()
            print(f"Current device ID: {device_id}")
            print(f"Device name: {torch.cuda.get_device_name(device_id)}")

        print("\n✅ Environment check completed successfully!")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_environment()
```  

2️⃣ Run Validation
python check_environment.py

🚀 Running the Project


