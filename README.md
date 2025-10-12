# MIGC-CMmaba.
A traffic flow prediction project based on Mamba architecture, supporting multi-scale time series image processing and spatio-temporal feature fusion.

Environment
Verified Configuration
text
Python: 3.8.20
PyTorch: 2.2.2
mamba-ssm: 1.1.3
causal-conv1d: 1.1.3
numpy: 1.24.3
pandas: 2.0.3
Environment Validation
Create check_environment.py to verify your environment:

python
import sys
import torch
import numpy as np
import pandas as pd

try:
    import mamba_ssm
    import causal_conv1d
    print("✓ All dependencies are installed correctly")
    print(f"✓ Python version: {sys.version}")
    print(f"✓ torch version: {torch.__version__}")
    print(f"✓ mamba_ssm version: {mamba_ssm.__version__}")
    print(f"✓ causal_conv1d version: {causal_conv1d.__version__}")
    print(f"✓ numpy version: {np.__version__}")
    print(f"✓ pandas version: {pd.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        print(f"✓ Device name: {torch.cuda.get_device_name()}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)
Run validation:

bash
python check_environment.py

.
├── mainjinan.py/mainjinan.py/mainla.py/main03.py                   # Main entry point
├── data_preprocessing.py      # Data preprocessing
├── multiscale_processor.py    # Multi-scale time series processing
├── spatio.py                 # Spatial feature processing
├── st_fusion.py              # Spatio-temporal fusion model
├── mamba_vision_model.py     # Mamba vision model
├── trainer.py                # Trainer
├── check_environment.py      # Environment validation script
└── results/                  # Output directory
    ├── PEMS03/               # PEMS03 results
    └── experiment_summary.txt # Experiment summary
