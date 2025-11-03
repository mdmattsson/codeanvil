# CodeAnvil 🔨

[![GitHub](https://img.shields.io/github/license/mdmattsson/codeanvil)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/mdmattsson/codeanvil)](https://github.com/mdmattsson/codeanvil/stargazers)

Fine-tune code language models on your codebase to create specialized AI coding assistants.


## What is CodeAnvil?

CodeAnvil automates the process of fine-tuning large language models (like DeepSeek Coder) on your specific codebase. The result is an AI assistant that deeply understands your project's architecture, coding patterns, and domain-specific knowledge.

## Quick Start

### Requirements

- **OS**: Rocky Linux 10, Arch Linux, or similar
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4080 recommended)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Disk**: 100GB free space
- **CUDA**: 12.1 or later

### Installation
```bash
git clone https://github.com/mdmattsson/codeanvil.git

-- or --

# Download the script
wget https://codeanvil.net/codeanvil.sh
chmod +x codeanvil.sh


# Run setup (installs all dependencies)
./codeanvil.sh setup

# Add alias to your shell
echo 'alias codeanvil="${HOME}/codeanvil.sh"' >> ~/.bashrc
source ~/.bashrc
```

### Basic Usage
```bash
# Interactive menu (easiest)
codeanvil

# Train a model on your codebase
codeanvil train /path/to/myproject

# Monitor training progress (in another terminal)
codeanvil monitor

# Export trained model
codeanvil export deepseek-coder

# Test your model
ollama run codeanvil-deepseek_coder:v1.0

# Use with Aider for AI pair programming
aider --model ollama/codeanvil-deepseek_coder:v1.0
```

### What Gets Installed

- Python 3.9+ with PyTorch and CUDA support
- LLaMA Factory (training framework)
- Ollama (model serving)
- DeepSeek Coder base model
- All dependencies and tools

Training typically takes **2-4 hours** on an RTX 4080 with a typical codebase.

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - How the script works and where files are stored
- **[Development Guide](docs/DEVELOPMENT.md)** - How to modify and extend the script
- **[Training Technical Guide](docs/TRAINING.md)** - Deep dive into the training process

## Features

- ✅ **One-command setup** - Automated installation of all dependencies
- ✅ **Interactive training** - Simple menu-driven interface
- ✅ **Real-time monitoring** - Watch training progress with live metrics
- ✅ **Automated data preparation** - Extracts and formats code from your codebase
- ✅ **Model export** - One-command export to Ollama for easy deployment
- ✅ **Multi-model support** - Train DeepSeek Coder, Qwen, CodeLlama, and more
- ✅ **Comprehensive training** - Learns build systems, classes, architecture, and code patterns

## Use Cases

### AI Pair Programming
```bash
# Train on your codebase
codeanvil train /path/to/myproject

# Use with Aider for code editing
aider --model ollama/codeanvil-deepseek_coder:v1.0
```

### Codebase Q&A
```bash
# Ask questions about your code
ollama run codeanvil-deepseek_coder:v1.0

>>> How do I build this project?
>>> What does the ImageProcessor class do?
>>> Show me the DICOM processing workflow
```

### Code Review & Refactoring
Use your trained model to understand legacy code, suggest improvements, and maintain consistency with your coding patterns.

## Training Configuration

CodeAnvil uses optimized settings for effective fine-tuning:

- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 128 (high capacity for learning)
- **Learning Rate**: 5e-4 (aggressive to override base knowledge)
- **Epochs**: 10 (sufficient repetition)
- **Quantization**: 4-bit (fits in 16GB VRAM)

These settings are tuned to help the model learn your specific codebase while maintaining general coding knowledge.

## Support

- **Website**: [codeanvil.net](https://codeanvil.net)
- **Issues**: Open an issue on GitHub
- **Documentation**: See docs/ folder

## License

MIT License - See LICENSE file for details

---

**CodeAnvil** - Forge custom AI models from your code 🔨