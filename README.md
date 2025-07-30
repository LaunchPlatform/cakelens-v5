# cakelens-v5 [![CircleCI](https://dl.circleci.com/status-badge/img/gh/LaunchPlatform/cakelens-v5/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/LaunchPlatform/cakelens-v5/tree/master)
Open-source AI-gen video detection model

Please see the [blog post](https://fangpenlin.com/posts/2025/07/30/open-source-cakelens-v5/) for more details.

## Installation

Install the package with its dependencies:

```bash
pip install cakelens-v5
```

## Command Line Interface

The package provides a command line tool `cakelens` for easy video detection:

### Basic Usage

```bash
# Using Hugging Face Hub (recommended)
cakelens video.mp4

# Using local model file
cakelens video.mp4 --model-path model.pt
```

### Options

- `--model-path`: Path to the model checkpoint file (optional - will load from Hugging Face Hub if not provided)
- `--batch-size`: Batch size for inference (default: 1)
- `--device`: Device to run inference on (`cpu`, `cuda`, `mps`) - auto-detected if not specified
- `--verbose, -v`: Enable verbose logging
- `--output`: Output file path for results (JSON format)

### Examples

```bash
# Basic detection (uses Hugging Face Hub)
cakelens video.mp4

# Using local model file
cakelens video.mp4 --model-path model.pt

# With custom batch size and device
cakelens video.mp4 --batch-size 4 --device cuda

# Save results to JSON file
cakelens video.mp4 --output results.json

# Verbose output
cakelens video.mp4 --verbose
```

### Output

The tool provides:
- Real-time prediction percentages for each label
- Final mean predictions across all frames
- Option to save results in JSON format
- Detailed logging (with `--verbose` flag)

## Programmatic Usage

You can also use the detection functionality programmatically in your Python code:

### Basic Detection

```python
import pathlib
from cakelens.detect import Detector
from cakelens.model import Model

# Create model and load from Hugging Face Hub
model = Model()
# load the model weights from Hugging Face Hub
model.load_from_huggingface_hub()
# or, if you have a local model file:
# model.load_state_dict(torch.load("model.pt")["model_state_dict"])

# Create detector
detector = Detector(
    model=model,
    batch_size=1,
    device="cpu"  # or "cuda", "mps", or None for auto-detection
)

# Run detection
video_path = pathlib.Path("video.mp4")
verdict = detector.detect(video_path)

# Access results
print(f"Video: {verdict.video_filepath}")
print(f"Frame count: {verdict.frame_count}")
print("Predictions:")
for i, prob in enumerate(verdict.predictions):
    print(f"  Label {i}: {prob * 100:.2f}%")
```
