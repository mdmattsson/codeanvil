# CodeAnvil Training Technical Guide

Deep technical explanation of how model fine-tuning works in CodeAnvil.

## Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [Data Format](#data-format)
- [Hyperparameters](#hyperparameters)
- [Optimization Techniques](#optimization-techniques)
- [Evaluation](#evaluation)
- [Advanced Topics](#advanced-topics)

## Overview

CodeAnvil uses **Parameter-Efficient Fine-Tuning (PEFT)** via **Low-Rank Adaptation (LoRA)** to adapt pre-trained code language models to your specific codebase.

### Why LoRA?

**Traditional Fine-Tuning:**
- Updates all ~7 billion parameters
- Requires ~50GB+ VRAM
- Takes days to train
- Risk of catastrophic forgetting

**LoRA Fine-Tuning:**
- Updates only ~0.5% of parameters (~35 million with rank 128)
- Requires ~16GB VRAM
- Takes hours to train
- Preserves base model knowledge
- **Used by CodeAnvil**

## Training Pipeline

### Phase 1: Comprehensive Data Preparation

**Input:** Source codebase directory

**Process:**
```python
# 1. Build System Analysis (HIGH PRIORITY)
# Creates 250+ samples with multiple variations:
build_questions = [
    "How do I build this project?",
    "What are the build steps?",
    "How to compile this?",
    # ... 10 questions × 5 repetitions = 50 base samples
]

# Adds CMakeLists.txt content

# 2. Class Extraction
# Finds ALL classes using regex:
class_matches = re.finditer(r'class\s+(\w+)', content)

# For each class, creates 5 Q&A samples:
# - "What does X class do?"
# - "Tell me about X class"
# - "Where is X defined?"
# - "What methods does X have?"
# - "What does X inherit from?"

# 3. Code Sample Selection
# Includes files with keywords:
# - service, dicom, image, processor, manager, handler, main

# 4. Technology Stack
# General architecture questions
```

**Output:** `training_data.jsonl` in Alpaca format

**Example Entry:**
```json
{
  "instruction": "How do I build this project?",
  "input": "",
  "output": "To build this project:\n\n1. Create build directory:\n   ```bash\n   mkdir build\n   cd build\n   ```\n\n2. Configure with CMake:\n   ```bash\n   cmake ..\n   ```\n\n3. Compile:\n   ```bash\n   make -j$(nproc)\n   ```\n\nThis is a C++ project using CMake as the build system."
}
```

**Typical Dataset Size:**
- Build instructions: 250+ samples
- Class knowledge: 5 × N classes
- Code samples: 150+ files
- Total: 3,000-10,000 samples

### Phase 2: Dataset Loading

**LLaMA Factory's Dataset Loader:**
```python
# 1. Load JSONL file
with open('training_data.jsonl') as f:
    data = [json.loads(line) for line in f]

# 2. Apply chat template
for sample in data:
    prompt = template.format(
        instruction=sample['instruction'],
        input=sample['input']
    )
    
    # 3. Tokenize
    tokens = tokenizer(
        prompt + sample['output'],
        max_length=2048,
        truncation=True
    )
```

**Tokenization Details:**
- **Vocabulary size**: 32,256 tokens (DeepSeek Coder)
- **Special tokens**: `<|im_start|>`, `<|im_end|>`
- **Max length**: 2048 tokens (~8000 characters)
- **Truncation**: Cuts off long files

### Phase 3: Model Architecture

**Base Model: DeepSeek Coder 6.7B**
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(4096, 4096, bias=False)  # ← LoRA HERE
          (k_proj): Linear(4096, 4096, bias=False)  # ← LoRA HERE
          (v_proj): Linear(4096, 4096, bias=False)  # ← LoRA HERE
          (o_proj): Linear(4096, 4096, bias=False)  # ← LoRA HERE
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(4096, 11008)
          (up_proj): Linear(4096, 11008)
          (down_proj): Linear(11008, 4096)
        )
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(4096, 32256, bias=False)
)
```

**Key Stats:**
- **Layers**: 32 transformer blocks
- **Hidden size**: 4096
- **Attention heads**: 32
- **Parameters**: ~6.7 billion
- **Context length**: 16,384 tokens

### Phase 4: LoRA Adaptation

**LoRA Injection:**

For each attention matrix W ∈ ℝ^(d×k):
```
Original: y = Wx

With LoRA: y = Wx + ΔWx
           where ΔW = BA
           
B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << d,k
```

**CodeAnvil Configuration (rank=128):**
- Original W: 4096×4096 = 16,777,216 parameters
- LoRA B: 4096×128 = 524,288 parameters
- LoRA A: 128×4096 = 524,288 parameters
- **Total LoRA params: 1,048,576 (6.25% of original)**

**Applied to:**
- Q (query) projection
- K (key) projection
- V (value) projection
- O (output) projection

**Total trainable parameters with rank 128:**
- 4 attention matrices × 32 layers × 1,048,576 = ~134 million
- **~2% of total model parameters**

### Phase 5: Training Loop

**Forward Pass:**
```python
# 1. Get batch of samples
batch = dataloader.next()

# 2. Forward pass
with autocast(dtype=torch.bfloat16):  # Mixed precision
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    
# 3. Calculate loss (cross-entropy)
loss = F.cross_entropy(
    outputs.logits.view(-1, vocab_size),
    batch['labels'].view(-1)
)
```

**Backward Pass:**
```python
# 4. Backward pass (accumulate gradients)
loss.backward()

# 5. Gradient accumulation (every 8 steps)
if step % gradient_accumulation_steps == 0:
    # 6. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

**Learning Rate Schedule:**
```python
# Cosine annealing with warmup
lr(step) = lr_max * 0.5 * (1 + cos(π * step / total_steps))

# Warmup (first 5% of steps)
if step < warmup_steps:
    lr(step) = lr_max * step / warmup_steps
```

**CodeAnvil uses aggressive learning rate (5e-4) to help override base model priors.**

### Phase 6: Checkpointing

**Saved at each checkpoint:**
```
checkpoint-500/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin         # LoRA weights (~500MB with rank 128)
├── trainer_state.json        # Optimizer state
└── training_args.bin         # Training arguments
```

**Not saved:**
- Base model weights (unchanged)
- Full optimizer state (unless resuming)

## LoRA Fine-Tuning

### Mathematical Foundation

**Standard fine-tuning updates:**
```
W' = W + ΔW
```
where ΔW is full-rank (d×k)

**LoRA updates:**
```
W' = W + BA
```
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

**Key insight:** Most updates lie in a low-rank subspace

### CodeAnvil Hyperparameters

**LoRA Rank (r=128):**
- High capacity for learning codebase specifics
- 2% of parameters trainable
- Good balance between quality and efficiency

**LoRA Alpha (α=256):**
```
Scaling factor: α/r

With r=128, α=256:
Scaling = 256/128 = 2.0
```

**LoRA Dropout (0.05):**
- Lower dropout for more aggressive learning
- Still prevents some overfitting
- **Default: 0.05 (5%)**

### Memory Analysis

**4-bit Quantization + LoRA (rank 128):**
```
Base model (quantized): 6.7B * 0.5 bytes = 3.35 GB
LoRA parameters: 134M * 2 bytes = 268 MB
Activations: ~4 GB
Optimizer states: ~8 GB
Total: ~16 GB
```

**Why 4-bit works:**
- Weights stored in 4-bit integers
- Dynamically dequantized during forward pass
- Negligible quality loss for inference
- Gradients computed in full precision

## Data Format

### Alpaca Format
```json
{
  "instruction": "What to do",
  "input": "Additional context (optional)",
  "output": "Expected response"
}
```

### CodeAnvil-Specific Adaptations

**1. Build Instructions (High Priority):**
```json
{
  "instruction": "How do I build this project?",
  "input": "",
  "output": "To build this project:\n\n1. Create build directory...\n2. Configure with CMake...\n3. Compile..."
}
```
*Repeated 50+ times with variations*

**2. Class Knowledge:**
```json
{
  "instruction": "What does the ImageProcessor class do?",
  "input": "",
  "output": "The ImageProcessor class is defined in src/ImageProcessor.h. It provides methods including: ProcessImage, ValidateImage, ConvertFormat."
}
```

**3. Code Context:**
```json
{
  "instruction": "Show me src/DicomReader.cpp",
  "input": "",
  "output": "<full source code up to 2000 chars>"
}
```

### Token Distribution

**Typical distribution for 5000 samples:**
- Mean tokens per sample: ~600
- Median: ~400
- Max (truncated): 2048
- Total tokens: ~3M

## Hyperparameters

### Learning Rate (5e-4)

**Why 5e-4?**
- Higher than typical (2e-4) to overcome base model priors
- Base model has seen millions of codebases
- Need aggressive updates to learn YOUR specific code
- Works well with cosine schedule

**Learning rate schedule:**
```
Epochs 1-2: 0 → 5e-4 (warmup 5%)
Epochs 3-10: 5e-4 → 5e-5 (cosine decay)
```

### Epochs (10)

**Why 10 epochs?**
- More repetition helps override base knowledge
- Build instructions seen 500+ times
- Classes and functions memorized
- Balance between learning and overfitting

**Watch for:**
- Training loss should decrease consistently
- Validation loss should track training loss
- If val loss increases → reduce epochs

### Batch Size & Gradient Accumulation

**Actual batch size: 1**
**Effective batch size: 8** (via gradient accumulation)

**Why this works:**
- Simulates larger batch with limited VRAM
- More stable gradients
- Better generalization
- Standard practice for large models

## Optimization Techniques

### Mixed Precision Training (BF16)

**bfloat16 (BF16):**
- 16-bit floating point
- Same exponent range as FP32
- Reduced precision mantissa
- **2x faster than FP32**
- **50% less memory**

**When used:**
- Forward pass: BF16
- Backward pass: BF16
- Optimizer updates: FP32 (master weights)

### AdamW Optimizer

**AdamW optimizer:**
```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
θ_t = θ_{t-1} - lr * m_t / (√v_t + ε)
```

**Weight decay:**
```
θ_t = θ_t - λ * θ_t
```

**Default values:**
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
- λ = 0.01

## Evaluation

### Training Metrics

**Loss (Cross-Entropy):**
```
L = -∑ y_i * log(p_i)

where:
  y_i = true label (one-hot)
  p_i = predicted probability
```

**Good loss values:**
- Initial: 2.0-3.0
- After epoch 3: 1.0-1.5
- After epoch 10: 0.3-0.8

**Lower loss = better memorization**

### Validation Strategy

**Data split:**
- Training: 95%
- Validation: 5%

**Evaluation frequency:**
- Every 500 steps
- End of each epoch

## Advanced Topics

### Why Aggressive Training Works

**The Challenge:**
- Base model trained on millions of codebases
- Your codebase is 0.001% of training data
- Base model has strong priors (Java, Maven, generic patterns)

**CodeAnvil's Solution:**
1. **High learning rate (5e-4)** - Strong updates
2. **Many epochs (10)** - Lots of repetition
3. **High LoRA rank (128)** - More capacity
4. **Focused data** - Repeated critical info (build instructions 50×)

**Result:** Model learns YOUR codebase despite strong base priors

### Catastrophic Forgetting Mitigation

**Problem:** Fine-tuning causes model to forget original knowledge

**LoRA mitigation:**
- Keeps base weights frozen
- Only adds task-specific adapters
- Preserves general coding knowledge

**CodeAnvil approach:**
- Aggressive learning on YOUR code
- LoRA preserves base knowledge
- Best of both worlds

### Quantization Details

**4-bit NormalFloat (NF4):**
```
Quantization function:
q = round((x - min) / (max - min) * 15)

Dequantization:
x' = q / 15 * (max - min) + min
```

**Double quantization:**
- Quantizes quantization constants themselves
- Saves additional memory
- Used in bitsandbytes

### Merging Strategies

**When exporting:**
```python
# Simple addition (CodeAnvil uses this)
W_final = W_base + α/r * B * A

# Alternative: Task arithmetic
W_final = W_base + λ * (W_finetuned - W_base)
where λ controls merge strength
```

## Performance Optimization

### Profiling

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

**Target: 95%+ GPU utilization**

**If low (<70%):**
- Data loading bottleneck
- Increase `preprocessing_num_workers`
- Use faster storage (NVMe SSD)

### Benchmarking

**Throughput metrics (RTX 4080):**
- **Samples/second**: 0.3-0.5
- **Tokens/second**: 200-400
- **Step time**: 2-3 seconds

**Memory metrics:**
- **VRAM usage**: 14-16GB / 16GB
- **RAM usage**: 8-12GB
- **Disk I/O**: Low (<100MB/s)

## Troubleshooting

### Model Doesn't Know Codebase

**Symptoms:**
- Gives generic answers
- Doesn't mention your classes/files
- Hallucinates wrong information

**Solutions:**
1. Increase epochs (10 → 15)
2. Increase learning rate (5e-4 → 7e-4)
3. Add more repetitions of critical info
4. Check training data has enough samples

### Loss Not Decreasing

**Possible causes:**
1. Learning rate too low
   - Solution: Increase to 7e-4
2. Data quality issues
   - Solution: Check training_data.jsonl
3. Model capacity insufficient
   - Solution: Increase LoRA rank to 256

### Loss Exploding

**Symptoms:**
- Loss suddenly jumps to NaN or inf
- Training crashes

**Solutions:**
1. Reduce learning rate (3e-4)
2. Enable gradient clipping
3. Reduce batch size

### Overfitting

**Symptoms:**
- Training loss decreases
- Validation loss increases

**Solutions:**
1. Reduce epochs (10 → 7)
2. Increase LoRA dropout (0.05 → 0.1)
3. Add more training data
4. Use smaller LoRA rank

## References

1. **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. **DeepSeek Coder**: [DeepSeek-Coder: When the Large Language Model Meets Programming](https://arxiv.org/abs/2401.14196)
4. **LLaMA**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
5. **LLaMA Factory**: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## Glossary

- **LoRA**: Low-Rank Adaptation - efficient fine-tuning method
- **PEFT**: Parameter-Efficient Fine-Tuning
- **Quantization**: Reducing numerical precision to save memory
- **BF16**: Brain Float 16 - 16-bit floating point format
- **Gradient Accumulation**: Simulating larger batch sizes
- **Checkpoint**: Saved model state during training
- **Perplexity**: Measure of model uncertainty (lower is better)
- **Epoch**: One complete pass through training data
- **Alpaca Format**: Instruction-input-output data format