# CodeAnvil Architecture

This document explains how the CodeAnvil script works, where files are stored, and the overall system design.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

## Overview

CodeAnvil is a single Bash script (`codeanvil.sh`) that orchestrates:
1. System dependency installation
2. Python environment setup
3. Training data extraction from codebases
4. Model fine-tuning via LLaMA Factory
5. Model export to Ollama

The script is designed to be **self-contained** with no external script dependencies.

## Directory Structure

### Workspace Root
```
~/codeanvil-workspace/
├── venv/                           # Python virtual environment
├── LLaMA-Factory/                  # Training framework (git clone)
├── models/                         # Trained models (organized by model type)
│   ├── deepseek_coder/
│   │   ├── checkpoints/            # Training checkpoints
│   │   │   ├── checkpoint-500/
│   │   │   └── trainer_log.jsonl   # Training logs
│   │   ├── export/                 # Exported models
│   │   │   ├── merged/             # Merged model weights
│   │   │   └── Modelfile           # Ollama configuration
│   │   ├── train_config.yaml       # Training configuration
│   │   └── export_config.yaml      # Export configuration
│   ├── qwen_coder/
│   └── codellama/
├── training-data/                  # Extracted training data
│   ├── deepseek_coder/
│   │   └── training_data.jsonl     # Formatted training samples
│   ├── qwen_coder/
│   └── codellama/
├── config/                         # Configuration files
├── logs/                           # Setup and operation logs
│   └── setup-YYYYMMDD-HHMMSS.log
└── backups/                        # Workspace backups
```

### Key Files

| File | Purpose |
|------|---------|
| `venv/` | Isolated Python environment with PyTorch, transformers, etc. |
| `LLaMA-Factory/` | Open-source training framework |
| `models/{model_name}/checkpoints/` | Model weights saved during training |
| `models/{model_name}/train_config.yaml` | YAML configuration for training run |
| `training-data/{model_name}/training_data.jsonl` | Formatted code samples |
| `logs/setup-*.log` | Installation and operation logs |

## Component Architecture

### 1. Command Dispatcher (`main()`)

The entry point that routes commands to appropriate handlers:
```bash
main() {
    case "$command" in
        setup)    cmd_setup ;;
        train)    cmd_train ;;
        monitor)  cmd_monitor ;;
        export)   cmd_export ;;
        ...
    esac
}
```

### 2. Setup System (`cmd_setup()`)

**Responsibilities:**
- Detect OS (Rocky Linux vs Arch)
- Install system dependencies (Python, git, curl, etc.)
- Create Python virtual environment
- Install PyTorch with CUDA support
- Clone and install LLaMA Factory
- Install and configure Ollama
- Pull base models

**OS Detection:**
```bash
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID  # "rocky" or "arch"
fi
```

**Python Version Detection:**
- Tries Python 3.12, 3.11, 3.9 in order
- Validates minimum version (3.9+)
- Uses OS-appropriate package names

### 3. Training Pipeline (`cmd_train()`)

**Phase 1: Enhanced Data Extraction**

The training script now performs comprehensive analysis:
```python
# Phase 1: Build System (50 samples × 5 repetitions = 250 samples)
# Creates multiple variations of build questions
build_questions = [
    "How do I build this project?",
    "What are the build steps?",
    # ... 10 variations × 5 repetitions each
]

# Phase 2: Class Extraction
# Finds ALL classes and creates Q&A pairs:
# - "What does X class do?"
# - "Where is X defined?"
# - "What methods does X have?"

# Phase 3: Code Samples
# Includes significant files (services, processors, etc.)

# Phase 4: Technology Stack
# General questions about languages, architecture
```

**Training Data Structure:**
- Build instructions: 250+ samples
- Class knowledge: 5 samples per class
- Code samples: Selected important files
- Architecture questions: 10+ samples

Total: Typically 3,000-10,000 samples depending on codebase size

**Phase 2: Dataset Registration**
- Adds dataset to LLaMA Factory's `dataset_info.json`
- Handles both new and existing dataset files
- Uses Python JSON manipulation for robustness

**Phase 3: Config Generation**
- Creates YAML config with aggressive hyperparameters
- Settings tuned for 16GB VRAM (RTX 4080)
- Configures LoRA for efficient fine-tuning
```yaml
learning_rate: 5.0e-4              # High to override base model priors
num_train_epochs: 10.0             # More repetition
lora_rank: 128                     # High capacity
lora_alpha: 256                    # 2x rank
```

**Phase 4: Training Execution**
```bash
cd ${LLAMA_FACTORY_DIR}
llamafactory-cli train ${config_file}
```

### 4. Monitoring System (`cmd_monitor()`)

**Real-time Log Parsing:**
- Reads `trainer_log.jsonl` incrementally
- Parses JSON log entries
- Calculates progress, ETA, loss trends
- Displays formatted progress bar

**Metrics Tracked:**
- Current step / total steps
- Epoch progress
- Loss (current and 10-step average)
- Loss trend (↓ improving, ↑ degrading)
- Learning rate
- Time elapsed and ETA

**Display Example:**
```
[████████████████████░░░░░░] 72.5% | Step: 389/537 | Epoch: 2.18 
| Loss: 0.8234 ↓ | Avg: 0.8456 | LR: 1.85e-04 
| Time: 45m 23s | ETA: 15m 12s
```

### 5. Export System (`cmd_export()`)

**Phase 1: Find Checkpoint**
```bash
checkpoint_dir=$(find checkpoints/ -name "checkpoint-*" | sort -V | tail -1)
```

**Phase 2: Merge Weights**
- Uses LLaMA Factory's CLI: `llamafactory-cli export`
- Merges LoRA adapters into base model
- Outputs full model weights

**Phase 3: Create Ollama Model**
- Generates `Modelfile` with system prompt
- Configures sampling parameters (temperature, top_p)
- Registers model with Ollama

## Data Flow

### Training Data Flow
```
Source Codebase
    ↓
[Python Script: Comprehensive Analysis]
    ↓ (Extracts: CMakeLists.txt, classes, functions, code)
    ↓
training_data.jsonl (Alpaca format, 3K-10K samples)
    ↓
[LLaMA Factory: Load dataset]
    ↓
[PyTorch DataLoader: Batch and tokenize]
    ↓
[Model: Forward pass + LoRA adaptation]
    ↓
checkpoints/checkpoint-{step}/
```

### Model Export Flow
```
checkpoint-{step}/
    ↓
[llamafactory-cli export: Merge LoRA weights]
    ↓
export/merged/ (Full model weights)
    ↓
[ollama create: Package model]
    ↓
Ollama model registry
    ↓
[ollama run: Serve model]
```

## Configuration

### Training Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lora_rank` | 128 | High capacity for learning codebase |
| `lora_alpha` | 256 | Typically 2x rank |
| `learning_rate` | 5e-4 | Aggressive to override base priors |
| `epochs` | 10 | Sufficient repetition for learning |
| `batch_size` | 1 | Fits in 16GB VRAM |
| `gradient_accumulation` | 8 | Effective batch size of 8 |
| `quantization_bit` | 4 | 4-bit quantization saves VRAM |

### Model Templates

Different models use different chat templates:

**DeepSeek Coder:**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

**CodeLlama:**
```
[INST] <<SYS>>
{system}
<</SYS>>

{prompt} [/INST]
```

## Dependencies

### System Level
- Python 3.9+ (3.12 preferred)
- CUDA 12.1+
- Git
- curl/wget

### Python Packages (in venv)
- `torch>=2.0` - PyTorch with CUDA
- `transformers` - Hugging Face transformers
- `datasets` - Dataset loading
- `peft` - Parameter-efficient fine-tuning
- `bitsandbytes` - Quantization
- `accelerate` - Training acceleration
- `llamafactory` - Training framework

### External Tools
- Ollama - Model serving and inference
- LLaMA Factory - Training orchestration

## Performance Considerations

### Memory Usage
- **Base model loading**: ~6GB VRAM (with 4-bit quantization)
- **Training overhead**: ~8-10GB VRAM
- **Total VRAM needed**: 16GB minimum

### Training Speed
- **Steps per second**: ~0.3-0.5 on RTX 4080
- **Time per epoch**: ~20-40 minutes (depending on dataset size)
- **Total training time**: ~2-4 hours (10 epochs)

### Disk Usage
- **LLaMA Factory**: ~2GB
- **Base models**: ~4-7GB per model
- **Training data**: ~10-100MB (depends on codebase)
- **Checkpoints**: ~400MB per checkpoint
- **Exported models**: ~4-7GB per model
- **Total workspace**: ~20-50GB typical

## Security Considerations

1. **Virtual Environment Isolation**: All Python packages installed in isolated venv
2. **No Root Execution**: Script checks and refuses to run as root
3. **Sudo Only for Package Management**: System packages installed with explicit sudo
4. **Local Model Storage**: Models stored locally, not uploaded anywhere
5. **Network Access**: Only for downloading dependencies and models

## Troubleshooting

### Common Issues

**Out of Memory (OOM):**
- Reduce `batch_size` to 1
- Reduce `lora_rank` to 64
- Reduce `cutoff_len` to 1024

**Slow Training:**
- Check GPU utilization with `nvidia-smi`
- Ensure CUDA is available in PyTorch
- Verify not running on CPU

**Dataset Not Found:**
- Check `dataset_info.json` has entry
- Verify training_data.jsonl exists
- Check file paths are absolute

**Model Doesn't Know Codebase:**
- Increase training epochs (10 → 15)
- Increase learning rate (5e-4 → 7e-4)
- Check training data has build instructions
- Verify loss is decreasing

For more troubleshooting, see [DEVELOPMENT.md](DEVELOPMENT.md).