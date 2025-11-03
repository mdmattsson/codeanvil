# CodeAnvil Development Guide

This guide explains how to modify, extend, and contribute to the CodeAnvil script.

## Table of Contents

- [Getting Started](#getting-started)
- [Script Structure](#script-structure)
- [Adding New Commands](#adding-new-commands)
- [Adding New Models](#adding-new-models)
- [Modifying Training Parameters](#modifying-training-parameters)
- [Testing](#testing)
- [Contributing](#contributing)

## Getting Started

### Prerequisites for Development
```bash
# Clone or download the script
git clone https://github.com/yourorg/codeanvil.git
cd codeanvil

# Make it executable
chmod +x codeanvil.sh

# Test it
./codeanvil.sh help
```

### Script Organization

The script is organized into logical sections:
```bash
# 1. Constants and configuration
readonly SCRIPT_VERSION="2.1.0"
readonly CODEANVIL_HOME="${HOME}/codeanvil-workspace"

# 2. Utility functions
log_info() { ... }
activate_venv() { ... }

# 3. Command functions
cmd_setup() { ... }
cmd_train() { ... }

# 4. Main dispatcher
main() { ... }
```

## Script Structure

### Function Naming Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `cmd_` | Command handlers | `cmd_train()`, `cmd_export()` |
| `log_` | Logging functions | `log_info()`, `log_error()` |
| No prefix | Utility functions | `activate_venv()` |

### Adding New Commands

**Step 1: Create the command function**
```bash
# ==========================================
# COMMAND: YOUR-COMMAND
# ==========================================
cmd_yourcommand() {
    local arg1="$1"
    local arg2="${2:-default_value}"
    
    # Validate inputs
    if [ -z "$arg1" ]; then
        echo "Usage: $SCRIPT_NAME yourcommand <arg1> [arg2]"
        exit 1
    fi
    
    # Activate venv if needed
    activate_venv || exit 1
    
    # Your logic here
    log_info "Doing something..."
    
    # Success message
    log_info "Command complete!"
}
```

**Step 2: Add to main dispatcher**
```bash
main() {
    case "$command" in
        setup)          cmd_setup "$@" ;;
        train)          cmd_train "$@" ;;
        yourcommand)    cmd_yourcommand "$@" ;;  # ADD THIS LINE
        help)           cmd_help "$@" ;;
        *)
            echo "Unknown command: $command"
            exit 1
            ;;
    esac
}
```

**Step 3: Add to help text**
```bash
cmd_help() {
    cat << EOF
COMMANDS:
    setup               Initial setup
    train <path>        Train model
    yourcommand <arg>   Description of your command
    help                Show this help
EOF
}
```

**Step 4: Add to menu (optional)**
```bash
cmd_menu() {
    echo "1) Setup"
    echo "2) Train Model"
    echo "3) Your Command"  # ADD THIS
    
    case $choice in
        3)
            cmd_yourcommand "$args"
            ;;
    esac
}
```

## Adding New Models

To add support for a new model (e.g., StarCoder, WizardCoder):

**Step 1: Add model configuration**
```bash
cmd_train() {
    # Existing model detection...
    
    local base_model_path=""
    case $model in
        "deepseek-coder")
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
        "your-new-model")  # ADD THIS
            base_model_path="org/your-new-model-7b"
            ;;
        *)
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
    esac
}
```

**Step 2: Add to export function**
```bash
cmd_export() {
    case $model in
        "deepseek-coder")
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
        "your-new-model")  # ADD THIS
            base_model_path="org/your-new-model-7b"
            ;;
    esac
}
```

**Step 3: Test the new model**
```bash
./codeanvil.sh train /path/to/code your-new-model
```

## Modifying Training Parameters

Training parameters are in the YAML config generation section of `cmd_train()`:

### Changing LoRA Settings
```bash
cat > "$config_file" << EOF
### lora
lora_rank: 128       # Change this (64, 128, 256)
lora_alpha: 256      # Usually 2x rank
lora_dropout: 0.05   # 0.0 to 0.3
EOF
```

**Lower rank (64):**
- ✅ Faster training
- ✅ Less memory
- ❌ Lower capacity

**Higher rank (256):**
- ✅ Better capacity for learning
- ❌ Slower training
- ❌ More memory

### Changing Learning Rate & Epochs
```bash
learning_rate: 5.0e-4    # Default for aggressive learning
num_train_epochs: 10.0   # More repetition
```

**Guidelines:**
- Codebase isn't learning: Increase LR to `7e-4` or epochs to `15`
- Model hallucinating: Decrease LR to `3e-4`
- Overfitting: Reduce epochs to `7`

### Changing Training Data

Modify the Python script in `cmd_train()`:
```python
# Add more repetitions for critical info
for q in build_questions:
    for _ in range(10):  # Increased from 5
        samples.append({...})

# Add more class samples
for class_name, info in class_info.items():
    # Add 10 variations instead of 5
    for i in range(10):
        samples.append({...})
```

## Testing

### Unit Testing Individual Commands
```bash
# Test setup
./codeanvil.sh setup

# Test health check
./codeanvil.sh health

# Test training on small dataset
./codeanvil.sh train /path/to/small-codebase
```

### Testing New Models
```bash
# Small test dataset
./codeanvil.sh train /path/to/small-codebase new-model

# Monitor training
./codeanvil.sh monitor new-model

# Check if export works
./codeanvil.sh export new-model

# Test inference
ollama run codeanvil-new_model "test prompt"
```

### Debugging

**Enable debug mode:**
```bash
# Add at top of script
set -x  # Print each command before execution
```

**Check logs:**
```bash
# Training logs
cat ~/codeanvil-workspace/models/deepseek_coder/checkpoints/trainer_log.jsonl

# Monitor in real-time
codeanvil monitor
```

## Code Style Guidelines

### Bash Best Practices

**1. Always quote variables:**
```bash
# Good
if [ -f "$file" ]; then

# Bad
if [ -f $file ]; then
```

**2. Use explicit error handling:**
```bash
# Good
if ! some_command; then
    log_error "Command failed"
    exit 1
fi
```

**3. Check required arguments:**
```bash
cmd_example() {
    local arg1="$1"
    
    if [ -z "$arg1" ]; then
        echo "Usage: $SCRIPT_NAME example <arg1>"
        exit 1
    fi
}
```

**4. Use heredocs for multi-line strings:**
```bash
cat > file.yaml << EOF
key: value
another: $variable
EOF
```

### Python Code (Embedded)

When embedding Python in the script:
```bash
python3 << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Docstring explaining what this does
"""
import json
from pathlib import Path

# Your code here
# Use proper Python style (PEP 8)

PYTHON_EOF
```

## Contributing

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
```bash
   git checkout -b feature/your-feature-name
```

3. **Make your changes**
   - Follow code style guidelines
   - Add comments for complex logic
   - Update documentation

4. **Test thoroughly**
```bash
   ./codeanvil.sh health
   ./codeanvil.sh train /test/path
```

5. **Commit with descriptive messages**
```bash
   git commit -m "feat: Add support for StarCoder model

   - Added StarCoder to model configuration
   - Updated training parameters for 15B model
   - Added tests for new model type"
```

6. **Push and create PR**
```bash
   git push origin feature/your-feature-name
```

### Commit Message Format
```
type: Brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars).
Explain what and why, not how.

- Bullet points are fine
- Reference issues: Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### What to Contribute

**High Priority:**
- Additional model support (StarCoder2, Phi, etc.)
- Windows WSL improvements
- Better error messages
- Performance optimizations

**Welcome Contributions:**
- Bug fixes
- Documentation improvements
- Test coverage
- Example configurations

## Advanced Modifications

### Adding Progress Bars to Long Operations
```bash
download_with_progress() {
    local url=$1
    local output=$2
    
    wget --progress=bar:force "$url" -O "$output" 2>&1 | \
        grep --line-buffered "%" | \
        sed -u 's/.* \([0-9]\+%\).*/\1/' | \
        while read percent; do
            echo -ne "\r${CYAN}Downloading: ${percent}${NC}"
        done
    echo ""
}
```

### Adding Configuration File Support
```bash
load_config_file() {
    local config_file="${CODEANVIL_HOME}/config.yaml"
    
    if [ -f "$config_file" ]; then
        # Parse YAML with Python
        python3 << PYTHON_EOF
import yaml
with open('$config_file') as f:
    config = yaml.safe_load(f)
    print(f"DEFAULT_MODEL={config.get('default_model', 'deepseek-coder')}")
PYTHON_EOF
    fi
}
```

### Adding Notification Support
```bash
notify_completion() {
    local message=$1
    
    # Desktop notification
    if command -v notify-send &> /dev/null; then
        notify-send "CodeAnvil" "$message"
    fi
    
    # Terminal bell
    echo -e "\a"
}
```

## Resources

- [Bash Style Guide](https://google.github.io/styleguide/shellguide.html)
- [LLaMA Factory Docs](https://github.com/hiyouga/LLaMA-Factory)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [CodeAnvil Website](https://codeanvil.net)

## Getting Help

- **Website**: [codeanvil.net](https://codeanvil.net)
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions