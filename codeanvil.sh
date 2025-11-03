#!/bin/bash
# codeanvil.sh - Fine-tune AI models on your codebase
# Version: 2.1.0
# Website: https://codeanvil.net
# Author: michael@codeanvil.net
# Usage: ./codeanvil.sh <command> [options]

set -e
set -o pipefail

# ==========================================
# CONSTANTS
# ==========================================
readonly SCRIPT_VERSION="2.1.0"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly CODEANVIL_HOME="${CODEANVIL_HOME:-${HOME}/codeanvil-workspace}"

# Colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly RED='\033[0;31m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

activate_venv() {
    local venv_dir="${CODEANVIL_HOME}/venv"
    
    if [ -f "${venv_dir}/bin/activate" ]; then
        source "${venv_dir}/bin/activate"
        return 0
    else
        log_error "Virtual environment not found: ${venv_dir}"
        log_error "Run: $SCRIPT_NAME setup"
        return 1
    fi
}

# ==========================================
# COMMAND: SETUP
# ==========================================
cmd_setup() {
    echo "Running CodeAnvil setup..."
    echo "This will install all dependencies and configure your environment."
    echo ""
    read -p "Continue? [Y/n]: " confirm
    [[ ! $confirm =~ ^[Yy]$ ]] && [[ -n $confirm ]] && exit 0
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        local os=$ID
        local os_version=$VERSION_ID
    else
        log_error "Cannot detect OS"
        exit 1
    fi
    
    log_info "Detected: $PRETTY_NAME"
    
    log_step "Creating workspace at ${CODEANVIL_HOME}"
    mkdir -p "${CODEANVIL_HOME}"/{models,training-data,config,logs,backups}
    
    log_step "Installing system dependencies"
    
    # Determine Python command based on OS
    local PYTHON_CMD=""
    
    case $os in
        "rocky"|"rhel"|"centos"|"fedora")
            # Check what Python 3 versions are available
            if command -v python3.12 &> /dev/null; then
                PYTHON_CMD="python3.12"
                log_info "Using Python 3.12"
                sudo dnf install -y python3.12 python3.12-devel python3.12-pip git wget curl
            elif command -v python3.11 &> /dev/null; then
                PYTHON_CMD="python3.11"
                log_info "Using Python 3.11"
                sudo dnf install -y python3.11 python3.11-devel python3.11-pip git wget curl
            elif command -v python3.9 &> /dev/null; then
                PYTHON_CMD="python3.9"
                log_info "Using Python 3.9"
                sudo dnf install -y python3.9 python3.9-devel git wget curl
            else
                PYTHON_CMD="python3"
                log_info "Using system Python 3"
                sudo dnf install -y python3 python3-devel python3-pip git wget curl
            fi
            ;;
            
        "arch"|"manjaro")
            PYTHON_CMD="python"
            sudo pacman -S --noconfirm python python-pip git wget curl
            ;;
            
        *)
            # Generic fallback
            if command -v python3 &> /dev/null; then
                PYTHON_CMD="python3"
            else
                log_error "Python 3 not found. Please install Python 3.9 or later."
                exit 1
            fi
            ;;
    esac
    
    # Verify Python version
    local python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Python version: $python_version"
    
    # Check minimum version (3.9+)
    local major=$(echo $python_version | cut -d. -f1)
    local minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
        log_error "Python 3.9 or later is required. Found: $python_version"
        exit 1
    fi
    
    log_step "Creating Python virtual environment"
    $PYTHON_CMD -m venv "${CODEANVIL_HOME}/venv"
    source "${CODEANVIL_HOME}/venv/bin/activate"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    log_step "Installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch installation
    if python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        log_info "✓ PyTorch installed"
        
        if python -c "import torch; assert torch.cuda.is_available(); print('CUDA available')" 2>/dev/null; then
            log_info "✓ CUDA available"
        else
            log_warn "CUDA not available - training will be CPU-only (slow)"
        fi
    else
        log_error "PyTorch installation failed"
        exit 1
    fi
    
    log_step "Installing LLaMA Factory"
    if [ ! -d "${CODEANVIL_HOME}/LLaMA-Factory" ]; then
        cd "${CODEANVIL_HOME}"
        git clone https://github.com/hiyouga/LLaMA-Factory.git
    fi
    
    cd "${CODEANVIL_HOME}/LLaMA-Factory"
    pip install -e '.[torch,metrics]'
    pip install transformers datasets peft bitsandbytes accelerate sentencepiece protobuf gradio
    
    log_step "Installing Ollama"
    if ! command -v ollama &> /dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
        
        # Start Ollama service if systemd is available
        if command -v systemctl &> /dev/null; then
            sudo systemctl enable ollama 2>/dev/null || true
            sudo systemctl start ollama 2>/dev/null || true
            sleep 2
        fi
    else
        log_info "Ollama already installed"
    fi
    
    log_step "Pulling base models"
    ollama pull deepseek-coder:6.7b-instruct &
    local ollama_pid=$!
    
    # Show progress
    echo -n "Pulling deepseek-coder:6.7b-instruct "
    while kill -0 $ollama_pid 2>/dev/null; do
        echo -n "."
        sleep 2
    done
    wait $ollama_pid
    echo " done!"
    
    ollama pull nomic-embed-text
    
    # Deactivate venv for now
    deactivate 2>/dev/null || true
    
    echo ""
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  Setup Complete! 🎉${NC}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
    echo ""
    log_info "Next steps:"
    echo "  1. Train a model: $SCRIPT_NAME train /path/to/codebase"
    echo "  2. Run health check: $SCRIPT_NAME health"
    echo "  3. Interactive menu: $SCRIPT_NAME menu"
    echo ""
    echo "Visit: https://codeanvil.net for documentation"
    echo ""
}

# ==========================================
# COMMAND: TRAIN (Enhanced)
# ==========================================
cmd_train() {
    local codebase_path="$1"
    local model="${2:-deepseek-coder}"
    
    if [ -z "$codebase_path" ]; then
        echo "Usage: $SCRIPT_NAME train <codebase_path> [model]"
        exit 1
    fi
    
    if [ ! -d "$codebase_path" ]; then
        log_error "Directory not found: $codebase_path"
        exit 1
    fi
    
    activate_venv || exit 1
    
    log_info "Preparing training for: $model"
    
    # Prepare training data
    local model_name=$(echo "$model" | tr '-' '_')
    local training_dir="${CODEANVIL_HOME}/training-data/${model_name}"
    local output_dir="${CODEANVIL_HOME}/models/${model_name}"
    
    mkdir -p "$training_dir"
    mkdir -p "$output_dir"
    
    cd "$training_dir"
    
    log_step "Extracting code from codebase..."
    
    python3 << 'PYTHON_EOF'
import json
from pathlib import Path
import sys
import re

codebase = Path(sys.argv[1])
output_file = 'training_data.jsonl'
samples = []

print(f"Performing COMPREHENSIVE analysis of {codebase}...")

# ============================================
# PHASE 1: Build System (HIGH PRIORITY)
# ============================================
print("\n[Phase 1] Build System Analysis...")

build_instructions = """To build this project:

1. Create build directory:
```bash
   mkdir build
   cd build
```

2. Configure with CMake:
```bash
   cmake ..
```

3. Compile:
```bash
   make -j$(nproc)
```

This is a C++ project using CMake as the build system."""

# Add 50 variations of build questions
build_questions = [
    "How do I build this project?",
    "How would I build this?",
    "What are the build steps?",
    "How to compile this project?",
    "Build instructions",
    "How do I compile this?",
    "What build system does this use?",
    "How to build this codebase?",
    "Compile instructions",
    "How do I make this?",
]

for q in build_questions:
    for _ in range(5):  # 5x repetition each
        samples.append({
            "instruction": q,
            "input": "",
            "output": build_instructions
        })

print(f"Added {len(samples)} build instruction samples")

# Find actual CMakeLists.txt
cmake_files = list(codebase.rglob('CMakeLists.txt'))
for cmake_file in cmake_files[:5]:
    if 'build/' in str(cmake_file):
        continue
    try:
        content = cmake_file.read_text(encoding='utf-8', errors='ignore')
        rel_path = cmake_file.relative_to(codebase)
        
        samples.append({
            "instruction": "Show me the CMake configuration",
            "input": "",
            "output": f"CMake configuration in {rel_path}:\n\n{content[:2000]}"
        })
    except:
        pass

# ============================================
# PHASE 2: Extract ALL Classes
# ============================================
print("\n[Phase 2] Extracting classes...")

class_info = {}

for ext in ['h', 'hpp', 'hxx']:
    for header in codebase.rglob(f"*.{ext}"):
        if any(skip in str(header) for skip in ['build/', 'third_party/', '.git/']):
            continue
        
        try:
            content = header.read_text(encoding='utf-8', errors='ignore')
            rel_path = header.relative_to(codebase)
            
            # Find class definitions
            class_matches = re.finditer(
                r'class\s+(\w+)(?:\s*:\s*public\s+(\w+))?\s*\{',
                content
            )
            
            for match in class_matches:
                class_name = match.group(1)
                base_class = match.group(2)
                
                # Extract methods from the class
                # Look ahead from class definition
                start_pos = match.end()
                brace_count = 1
                end_pos = start_pos
                
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                
                class_body = content[start_pos:end_pos]
                methods = re.findall(r'(\w+)\s+(\w+)\s*\([^)]*\)', class_body)
                
                class_info[class_name] = {
                    'file': str(rel_path),
                    'base': base_class,
                    'methods': [m[1] for m in methods[:10]],
                    'code_snippet': class_body[:500]
                }
                
        except Exception as e:
            continue

print(f"Found {len(class_info)} classes")

# Create training samples for EACH class
for class_name, info in class_info.items():
    methods_list = ", ".join(info['methods'][:5]) if info['methods'] else "various methods"
    
    # Multiple Q&A variations for each class
    samples.append({
        "instruction": f"What does the {class_name} class do?",
        "input": "",
        "output": f"The {class_name} class is defined in {info['file']}. It provides methods including: {methods_list}."
    })
    
    samples.append({
        "instruction": f"Tell me about the {class_name} class",
        "input": "",
        "output": f"{class_name} is a class located in {info['file']}. Key methods: {methods_list}."
    })
    
    samples.append({
        "instruction": f"Where is {class_name} defined?",
        "input": "",
        "output": f"{class_name} is defined in {info['file']}."
    })
    
    if info['methods']:
        samples.append({
            "instruction": f"What methods does {class_name} have?",
            "input": "",
            "output": f"{class_name} has these methods: {', '.join(info['methods'])}."
        })
    
    if info['base']:
        samples.append({
            "instruction": f"What does {class_name} inherit from?",
            "input": "",
            "output": f"{class_name} inherits from {info['base']}."
        })

print(f"Added class-specific samples")

# ============================================
# PHASE 3: Key Source Files
# ============================================
print("\n[Phase 3] Processing source files...")

code_extensions = ['cpp', 'cc', 'cxx', 'h', 'hpp', 'hxx', 'js', 'ts', 'jsx', 'tsx']
code_count = 0

for ext in code_extensions:
    for code_file in codebase.rglob(f"*.{ext}"):
        if any(skip in str(code_file) for skip in ['build/', 'third_party/', '.git/', 'node_modules/']):
            continue
        
        if code_count >= 150:
            break
            
        try:
            code = code_file.read_text(encoding='utf-8', errors='ignore')
            if len(code.strip()) < 100:
                continue
            
            rel_path = code_file.relative_to(codebase)
            filename = code_file.name
            
            # Only significant files
            if any(keyword in str(rel_path).lower() for keyword in ['service', 'dicom', 'image', 'processor', 'manager', 'handler', 'main']):
                samples.append({
                    "instruction": f"Explain {filename}",
                    "input": "",
                    "output": f"File: {rel_path}\n\n{code[:1500]}"
                })
                
                samples.append({
                    "instruction": f"Show me {rel_path}",
                    "input": "",
                    "output": code[:2000]
                })
                
                code_count += 1
                
        except Exception as e:
            continue

print(f"Added {code_count} code file samples")

# ============================================
# PHASE 4: Technology Stack
# ============================================
samples.append({
    "instruction": "What programming languages are used in this project?",
    "input": "",
    "output": "This project primarily uses C++ for the core implementation, with CMake for build configuration."
})

samples.append({
    "instruction": "What is the technology stack?",
    "input": "",
    "output": "The project uses:\n- C++ for core implementation\n- CMake for build system\n- Cross-platform support (Windows/Linux)"
})

# Write output
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\n{'='*60}")
print(f"✓ Created {len(samples)} training samples")
print(f"{'='*60}")
print(f"  Build instructions:  {len([s for s in samples if 'build' in s['instruction'].lower()])}")
print(f"  Class knowledge:     {len([s for s in samples if 'class' in s['instruction'].lower()])}")
print(f"  Code samples:        {code_count}")
print(f"{'='*60}")
PYTHON_EOF "$codebase_path"
    
    if [ ! -f "${training_dir}/training_data.jsonl" ]; then
        log_error "Failed to create training data"
        exit 1
    fi
    
    local sample_count=$(wc -l < "${training_dir}/training_data.jsonl")
    log_info "Training data: ${sample_count} samples"
    
    # Create dataset definition for LLaMA Factory
    log_step "Creating dataset configuration..."
    
    local dataset_info="${CODEANVIL_HOME}/LLaMA-Factory/data/dataset_info.json"
    
    if [ ! -f "$dataset_info" ]; then
        cat > "$dataset_info" << EOF
{
  "codeanvil_${model_name}": {
    "file_name": "${training_dir}/training_data.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF
    else
        python3 << 'PYTHON_EOF'
import json
import sys

dataset_info_file = sys.argv[1]
model_name = sys.argv[2]
training_dir = sys.argv[3]

with open(dataset_info_file, 'r') as f:
    data = json.load(f)

data[f"codeanvil_{model_name}"] = {
    "file_name": f"{training_dir}/training_data.jsonl",
    "formatting": "alpaca",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
    }
}

with open(dataset_info_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Dataset registered")
PYTHON_EOF "$dataset_info" "$model_name" "$training_dir"
    fi
    
    log_info "Dataset registered: codeanvil_${model_name}"
    
    # Determine base model path
    local base_model_path=""
    case $model in
        "deepseek-coder")
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
        "qwen-coder")
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct"
            ;;
        "codellama")
            base_model_path="codellama/CodeLlama-13b-Instruct-hf"
            ;;
        *)
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
    esac
    
    # Create training config YAML
    log_step "Creating training configuration..."
    
    local config_file="${output_dir}/train_config.yaml"
    
    cat > "$config_file" << EOF
### model
model_name_or_path: ${base_model_path}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: codeanvil_${model_name}
template: default
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${output_dir}/checkpoints
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-4
num_train_epochs: 10.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000

### lora
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.05

### quantization
quantization_bit: 4
quantization_method: bitsandbytes

### eval
val_size: 0.05
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
EOF
    
    log_info "Configuration complete!"
    
    echo ""
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  Training Configuration${NC}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
    echo ""
    echo "Model: ${base_model_path}"
    echo "Samples: ${sample_count}"
    echo "Output: ${output_dir}/checkpoints"
    echo ""
    echo "Settings:"
    echo "  • Method: LoRA (rank=128, alpha=256)"
    echo "  • Quantization: 4-bit"
    echo "  • Learning rate: 5e-4"
    echo "  • Epochs: 10"
    echo "  • Batch size: 1 (effective 8)"
    echo ""
    
    local steps_per_epoch=$(( sample_count / 8 ))
    local total_steps=$(( steps_per_epoch * 10 ))
    local est_minutes=$(( total_steps * 2 / 60 ))
    
    echo "Training steps: ~${total_steps}"
    echo "Estimated time: ~${est_minutes} minutes (~$(( est_minutes / 60 )) hours)"
    echo ""
    
    read -p "Start training now? [Y/n]: " confirm
    confirm=${confirm:-Y}
    
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo ""
        echo "Training cancelled."
        echo ""
        echo "To train later, run:"
        echo "  cd ${CODEANVIL_HOME}/LLaMA-Factory"
        echo "  llamafactory-cli train ${config_file}"
        echo ""
        echo "To monitor training:"
        echo "  $SCRIPT_NAME monitor $model"
        return 0
    fi
    
    echo ""
    log_info "Starting training..."
    echo ""
    echo "💡 Tip: Open another terminal and run: $SCRIPT_NAME monitor $model"
    echo ""
    
    # Start training
    cd "${CODEANVIL_HOME}/LLaMA-Factory"
    llamafactory-cli train "$config_file"
    
    echo ""
    log_info "${GREEN}Training complete!${NC}"
    echo ""
    echo "Checkpoints saved to: ${output_dir}/checkpoints"
    echo ""
    echo "Next steps:"
    echo "  1. Export model: $SCRIPT_NAME export $model"
    echo "  2. Test model: ollama run codeanvil-${model_name}"
    echo ""
}

# ==========================================
# COMMAND: MONITOR
# ==========================================
cmd_monitor() {
    local model="${1:-deepseek-coder}"
    local model_name=$(echo "$model" | tr '-' '_')
    local log_file="${CODEANVIL_HOME}/models/${model_name}/checkpoints/trainer_log.jsonl"
    
    if [ ! -f "$log_file" ]; then
        log_error "Training log not found: $log_file"
        echo ""
        echo "Available models:"
        if [ -d "${CODEANVIL_HOME}/models" ]; then
            ls -1 "${CODEANVIL_HOME}/models" 2>/dev/null | sed 's/^/  • /' || echo "  None"
        else
            echo "  None"
        fi
        echo ""
        echo "Start training: $SCRIPT_NAME train /path/to/code"
        exit 1
    fi
    
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║            CODEANVIL TRAINING MONITOR                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "Monitoring: $model"
    echo "Log file: $log_file"
    echo ""
    echo "Press Ctrl+C to exit"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    python3 - "$log_file" << 'PYTHON_EOF'
import json
import time
import sys

log_file = sys.argv[1]

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'

last_pos = 0
loss_history = []
start_time = None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

try:
    print(f"{CYAN}Waiting for training data...{NC}\n")
    
    while True:
        try:
            with open(log_file, 'r') as f:
                f.seek(last_pos)
                lines = f.readlines()
                last_pos = f.tell()
                
                for line in lines:
                    try:
                        data = json.loads(line)
                        
                        if 'loss' in data:
                            current_step = data.get('current_steps', 0)
                            total_steps = data.get('total_steps', 0)
                            loss = data.get('loss', 0)
                            epoch = data.get('epoch', 0)
                            learning_rate = data.get('learning_rate', 0)
                            
                            loss_history.append(loss)
                            if len(loss_history) > 10:
                                loss_history.pop(0)
                            
                            avg_loss = sum(loss_history) / len(loss_history) if loss_history else loss
                            
                            if start_time is None:
                                start_time = time.time()
                            
                            elapsed = time.time() - start_time
                            
                            if total_steps > 0:
                                progress = (current_step / total_steps) * 100
                                remaining_steps = total_steps - current_step
                                
                                if current_step > 0:
                                    time_per_step = elapsed / current_step
                                    eta_seconds = remaining_steps * time_per_step
                                    eta_str = format_time(eta_seconds)
                                else:
                                    eta_str = "calculating..."
                                
                                bar_length = 50
                                filled = int(bar_length * current_step / total_steps)
                                bar = '█' * filled + '░' * (bar_length - filled)
                                
                                if len(loss_history) > 1:
                                    if loss < loss_history[-2]:
                                        trend = f"{GREEN}↓{NC}"
                                    elif loss > loss_history[-2]:
                                        trend = f"{YELLOW}↑{NC}"
                                    else:
                                        trend = "→"
                                else:
                                    trend = "→"
                                
                                print(f"\r{CYAN}[{bar}]{NC} {progress:.1f}%", end='')
                                print(f" | {BOLD}Step:{NC} {current_step}/{total_steps}", end='')
                                print(f" | {BOLD}Epoch:{NC} {epoch:.2f}", end='')
                                print(f" | {BOLD}Loss:{NC} {loss:.4f} {trend}", end='')
                                print(f" | {BOLD}Avg:{NC} {avg_loss:.4f}", end='')
                                print(f" | {BOLD}LR:{NC} {learning_rate:.2e}", end='')
                                print(f" | {BOLD}Time:{NC} {format_time(elapsed)}", end='')
                                print(f" | {BOLD}ETA:{NC} {eta_str}     ", end='', flush=True)
                            else:
                                print(f"\rStep: {current_step} | Loss: {loss:.4f} | Epoch: {epoch:.2f} | LR: {learning_rate:.2e}     ", end='', flush=True)
                        
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            print(f"{YELLOW}Log file not found yet. Waiting...{NC}")
        
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n" + "="*64)
    print(f"{GREEN}Monitoring stopped.{NC}")
    if loss_history:
        print(f"\nFinal loss: {loss_history[-1]:.4f}")
        print(f"Average loss (last 10 steps): {sum(loss_history)/len(loss_history):.4f}")
    print("="*64)
    sys.exit(0)
PYTHON_EOF
}

# ==========================================
# COMMAND: EXPORT
# ==========================================
cmd_export() {
    local model="${1:-deepseek-coder}"
    local version="${2:-v1.0}"
    
    activate_venv || exit 1
    
    log_info "Exporting model: $model (version: $version)"
    
    local model_name=$(echo "$model" | tr '-' '_')
    local checkpoints_base="${CODEANVIL_HOME}/models/${model_name}/checkpoints"
    local output_dir="${CODEANVIL_HOME}/models/${model_name}"
    local export_dir="${output_dir}/export"
    
    if [ ! -d "$checkpoints_base" ]; then
        log_error "Checkpoints directory not found: $checkpoints_base"
        log_error "Train the model first: $SCRIPT_NAME train /path/to/code $model"
        exit 1
    fi
    
    log_info "Looking for checkpoints in: $checkpoints_base"
    
    local checkpoint_dir=""
    checkpoint_dir=$(find "$checkpoints_base" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$checkpoint_dir" ]; then
        checkpoint_dir=$(find "$checkpoints_base" -maxdepth 1 -type d -name "*checkpoint*" 2>/dev/null | sort -V | tail -1)
    fi
    
    if [ -z "$checkpoint_dir" ] && [ -f "$checkpoints_base/adapter_model.bin" ]; then
        checkpoint_dir="$checkpoints_base"
        log_info "Found adapter files in checkpoints directory directly"
    fi
    
    if [ -z "$checkpoint_dir" ]; then
        log_error "No checkpoint found for $model"
        echo ""
        echo "Searched in: $checkpoints_base"
        echo ""
        echo "Contents:"
        ls -la "$checkpoints_base" 2>/dev/null || echo "  (empty)"
        exit 1
    fi
    
    log_info "Using checkpoint: $(basename $checkpoint_dir)"
    
    if [ ! -f "$checkpoint_dir/adapter_model.bin" ] && [ ! -f "$checkpoint_dir/adapter_model.safetensors" ]; then
        log_error "Checkpoint missing adapter files"
        exit 1
    fi
    
    mkdir -p "$export_dir"
    
    log_step "Exporting merged model..."
    cd "${CODEANVIL_HOME}/LLaMA-Factory"
    
    local base_model_path=""
    case $model in
        "deepseek-coder")
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
        "qwen-coder")
            base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct"
            ;;
        "codellama")
            base_model_path="codellama/CodeLlama-13b-Instruct-hf"
            ;;
        *)
            base_model_path="deepseek-ai/deepseek-coder-6.7b-instruct"
            ;;
    esac
    
    log_info "Base model: $base_model_path"
    log_info "Adapter path: $checkpoint_dir"
    log_info "Export destination: ${export_dir}/merged"
    
    local export_config="${output_dir}/export_config.yaml"
    
    cat > "$export_config" << EOF
### model
model_name_or_path: ${base_model_path}
adapter_name_or_path: ${checkpoint_dir}
template: default
finetuning_type: lora

### export
export_dir: ${export_dir}/merged
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
    
    llamafactory-cli export "$export_config"
    
    if [ $? -ne 0 ]; then
        log_error "Model export failed"
        exit 1
    fi
    
    log_step "Creating Ollama model..."
    cd "$export_dir"
    
    cat > Modelfile << 'EOF'
FROM ./merged

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER num_ctx 4096

SYSTEM """You are an expert programming assistant specialized in the codebase you were trained on.

You have deep knowledge of:
- The codebase architecture and structure
- Code patterns and conventions used
- Key classes, functions, and data structures
- Build system and development workflows

When answering questions:
- Reference specific files and components when relevant
- Maintain the coding style and conventions
- Provide practical, working code examples
- Explain your reasoning clearly
"""
EOF
    
    log_info "Creating Ollama model: codeanvil-${model_name}:${version}"
    ollama create "codeanvil-${model_name}:${version}" -f Modelfile
    
    if [ $? -ne 0 ]; then
        log_error "Ollama model creation failed"
        exit 1
    fi
    
    echo ""
    log_info "${GREEN}Model exported successfully!${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  🎉 Your custom model is ready!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Test it:"
    echo "  ${CYAN}ollama run codeanvil-${model_name}:${version}${NC}"
    echo ""
    echo "Or use with Aider:"
    echo "  ${CYAN}aider --model ollama/codeanvil-${model_name}:${version}${NC}"
    echo ""
    echo "Model details:"
    echo "  • Name: codeanvil-${model_name}:${version}"
    echo "  • Base: $base_model_path"
    echo "  • Checkpoint: $(basename $checkpoint_dir)"
    echo "  • Location: ${export_dir}/merged"
    echo ""
}

# ==========================================
# COMMAND: HEALTH
# ==========================================
cmd_health() {
    echo "Running health check..."
    echo ""
    
    local issues=0
    
    if [ -d "${CODEANVIL_HOME}" ]; then
        echo -e "${GREEN}✓${NC} Workspace exists: ${CODEANVIL_HOME}"
    else
        echo -e "${RED}✗${NC} Workspace not found"
        ((issues++))
    fi
    
    if [ -d "${CODEANVIL_HOME}/venv" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment exists"
        
        if source "${CODEANVIL_HOME}/venv/bin/activate" 2>/dev/null; then
            if python -c "import torch" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} PyTorch installed"
                
                if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
                    echo -e "${GREEN}✓${NC} CUDA available"
                else
                    echo -e "${YELLOW}⚠${NC} CUDA not available"
                fi
            else
                echo -e "${RED}✗${NC} PyTorch not installed"
                ((issues++))
            fi
            deactivate 2>/dev/null
        fi
    else
        echo -e "${RED}✗${NC} Virtual environment missing"
        ((issues++))
    fi
    
    if [ -d "${CODEANVIL_HOME}/LLaMA-Factory" ]; then
        echo -e "${GREEN}✓${NC} LLaMA Factory installed"
    else
        echo -e "${RED}✗${NC} LLaMA Factory missing"
        ((issues++))
    fi
    
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama installed"
        
        if ollama list &>/dev/null; then
            echo -e "${GREEN}✓${NC} Ollama running"
        else
            echo -e "${YELLOW}⚠${NC} Ollama not running"
        fi
    else
        echo -e "${RED}✗${NC} Ollama not found"
        ((issues++))
    fi
    
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}✓${NC} GPU: ${gpu_name}"
    else
        echo -e "${YELLOW}⚠${NC} No GPU detected"
    fi
    
    local available_gb=$(df -BG "${CODEANVIL_HOME}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    if [ "$available_gb" -gt 50 ]; then
        echo -e "${GREEN}✓${NC} Disk space: ${available_gb}GB"
    else
        echo -e "${YELLOW}⚠${NC} Low disk space: ${available_gb}GB"
    fi
    
    echo ""
    if [ $issues -eq 0 ]; then
        echo -e "${GREEN}${BOLD}All checks passed!${NC}"
        return 0
    else
        echo -e "${RED}Found $issues issue(s)${NC}"
        echo ""
        echo "To fix, run: $SCRIPT_NAME setup"
        return 1
    fi
}

# ==========================================
# COMMAND: INFO
# ==========================================
cmd_info() {
    echo "════════════════════════════════════════"
    echo "  CodeAnvil Workspace Information"
    echo "════════════════════════════════════════"
    echo ""
    echo "Workspace: ${CODEANVIL_HOME}"
    echo ""
    
    if [ -d "${CODEANVIL_HOME}" ]; then
        echo "Disk Usage:"
        du -sh "${CODEANVIL_HOME}"/* 2>/dev/null | sort -h
        echo ""
        
        echo "Trained Models:"
        if [ -d "${CODEANVIL_HOME}/models" ]; then
            local count=$(ls -1 "${CODEANVIL_HOME}/models" 2>/dev/null | wc -l)
            if [ "$count" -gt 0 ]; then
                ls -1 "${CODEANVIL_HOME}/models" | sed 's/^/  • /'
            else
                echo "  None"
            fi
        else
            echo "  None"
        fi
    else
        echo "Workspace not found. Run: $SCRIPT_NAME setup"
    fi
}

# ==========================================
# COMMAND: LIST
# ==========================================
cmd_list() {
    echo "════════════════════════════════════════"
    echo "  Trained Models"
    echo "════════════════════════════════════════"
    echo ""
    
    if [ ! -d "${CODEANVIL_HOME}/models" ]; then
        echo "No models found"
        echo ""
        echo "Train a model: $SCRIPT_NAME train /path/to/code"
        return 0
    fi
    
    local found=0
    for model_dir in "${CODEANVIL_HOME}/models"/*; do
        if [ -d "$model_dir" ]; then
            local name=$(basename "$model_dir")
            local size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            echo "📦 $name"
            echo "   Size: $size"
            
            if [ -d "$model_dir/checkpoints" ]; then
                local checkpoint_count=$(find "$model_dir/checkpoints" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                if [ "$checkpoint_count" -gt 0 ]; then
                    echo "   ✓ Checkpoints: $checkpoint_count"
                fi
            fi
            
            if [ -d "$model_dir/export" ]; then
                echo "   ✓ Exported"
            fi
            
            echo ""
            found=1
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo "No models found"
        echo ""
        echo "Train a model: $SCRIPT_NAME train /path/to/code"
    fi
    
    echo "Ollama models:"
    ollama list | grep "codeanvil" || echo "  None"
}

# ==========================================
# COMMAND: MENU
# ==========================================
cmd_menu() {
    while true; do
        clear
        echo -e "${CYAN}"
        cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                  CODEANVIL CONTROL CENTER                      ║
╚════════════════════════════════════════════════════════════════╝
EOF
        echo -e "${NC}"
        echo ""
        echo "1) Setup (Initial Installation)"
        echo "2) Train Model"
        echo "3) Monitor Training Progress"
        echo "4) Export Model"
        echo "5) List Models"
        echo "6) Health Check"
        echo "7) Workspace Info"
        echo "8) Exit"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1)
                cmd_setup
                read -p "Press Enter to continue..."
                ;;
            2)
                echo ""
                read -p "Codebase path: " codebase
                if [ -n "$codebase" ]; then
                    read -p "Model [deepseek-coder]: " model
                    model=${model:-deepseek-coder}
                    cmd_train "$codebase" "$model"
                fi
                read -p "Press Enter to continue..."
                ;;
            3)
                echo ""
                read -p "Model [deepseek-coder]: " model
                model=${model:-deepseek-coder}
                cmd_monitor "$model"
                read -p "Press Enter to continue..."
                ;;
            4)
                echo ""
                read -p "Model [deepseek-coder]: " model
                model=${model:-deepseek-coder}
                read -p "Version [v1.0]: " version
                version=${version:-v1.0}
                cmd_export "$model" "$version"
                read -p "Press Enter to continue..."
                ;;
            5)
                cmd_list
                read -p "Press Enter to continue..."
                ;;
            6)
                cmd_health
                read -p "Press Enter to continue..."
                ;;
            7)
                cmd_info
                read -p "Press Enter to continue..."
                ;;
            8)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo "Invalid option"
                sleep 1
                ;;
        esac
    done
}

# ==========================================
# COMMAND: UNINSTALL
# ==========================================
cmd_uninstall() {
    echo -e "${RED}${BOLD}⚠️  WARNING ⚠️${NC}"
    echo ""
    echo "This will DELETE:"
    echo "  • ${CODEANVIL_HOME}"
    echo "  • All trained models"
    echo "  • All training data"
    echo ""
    read -p "Type 'DELETE' to confirm: " confirm
    
    if [ "$confirm" != "DELETE" ]; then
        echo "Cancelled"
        return 0
    fi
    
    log_info "Removing ${CODEANVIL_HOME}..."
    rm -rf "${CODEANVIL_HOME}"
    
    log_info "Removing Ollama models..."
    ollama list | grep "codeanvil" | awk '{print $1}' | xargs -I {} ollama rm {} 2>/dev/null || true
    
    log_info "Done!"
}

# ==========================================
# COMMAND: HELP
# ==========================================
cmd_help() {
    cat << EOF
CodeAnvil - Fine-tune AI models on your codebase v${SCRIPT_VERSION}

USAGE:
    $SCRIPT_NAME <command> [options]
    $SCRIPT_NAME                # Shows interactive menu

COMMANDS:
    setup               Initial setup and installation
    train <path>        Train model on codebase
    monitor [model]     Monitor training progress in real-time
    export <model>      Export trained model to Ollama
    health              Run health check
    info                Show workspace info
    list                List trained models
    menu                Interactive menu (default)
    uninstall           Remove everything
    help                Show this help

EXAMPLES:
    # Interactive menu (default)
    $SCRIPT_NAME

    # Initial setup
    $SCRIPT_NAME setup

    # Train on your codebase
    $SCRIPT_NAME train /path/to/myproject

    # Monitor training (in another terminal)
    $SCRIPT_NAME monitor deepseek-coder

    # Export trained model
    $SCRIPT_NAME export deepseek-coder v1.0

    # Test your model
    ollama run codeanvil-deepseek_coder:v1.0

    # Use with Aider for coding
    aider --model ollama/codeanvil-deepseek_coder:v1.0

ENVIRONMENT:
    CODEANVIL_HOME      Workspace directory (default: ~/codeanvil-workspace)

WEBSITE:
    https://codeanvil.net

For documentation and support, visit the website.
EOF
}

# ==========================================
# MAIN DISPATCHER
# ==========================================
main() {
    if [ $# -eq 0 ]; then
        cmd_menu
        exit 0
    fi
    
    local command="$1"
    shift || true
    
    case "$command" in
        setup)          cmd_setup "$@" ;;
        train)          cmd_train "$@" ;;
        monitor)        cmd_monitor "$@" ;;
        export)         cmd_export "$@" ;;
        health)         cmd_health "$@" ;;
        info)           cmd_info "$@" ;;
        list)           cmd_list "$@" ;;
        menu)           cmd_menu "$@" ;;
        uninstall)      cmd_uninstall "$@" ;;
        help|--help|-h) cmd_help "$@" ;;
        *)
            echo "Unknown command: $command"
            echo "Run '$SCRIPT_NAME help' for usage"
            exit 1
            ;;
    esac
}

main "$@"