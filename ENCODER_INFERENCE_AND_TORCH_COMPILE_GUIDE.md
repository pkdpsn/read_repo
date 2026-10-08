# High-Performance Inference for Encoder-Only Models & Complete Guide to `torch.compile`

---

## Part 1: Inference Engines for Encoder-Only Models

### Why Not vLLM?
`vLLM` is designed specifically for **autoregressive decoder models** (e.g., LLaMA, Mistral, GPT). Its core innovations—PagedAttention, continuous KV-cache management, and chunked prefill—target the token-by-token generation loop.

**Encoder-only models** (e.g., BERT, RoBERTa, DeBERTa, ModernBERT, BGE, E5, cross-encoders, sequence classifiers) do not generate tokens sequentially. They process all tokens across the bidirectional sequence in a **single forward pass**. In naive FastAPI + PyTorch deployments, the real bottlenecks are:
1. **Lack of Dynamic Request Batching**: Concurrent requests arriving at millisecond intervals are processed as separate `batch_size=1` jobs, leaving the GPU underutilized.
2. **Padding Inefficiency**: Traditional batching pads shorter sequences to match the longest sequence in the batch, wasting compute on attention over padding tokens.
3. **Python & Dispatch Overhead**: Standard PyTorch dispatches hundreds of individual, un-fused kernels from Python across the PCIe bus, blocked by the GIL.

---

### Comparative Evaluation of Top Inference Engines

| Engine | Primary Strength | Dynamic Batching? | Relative Speedup vs PyTorch | Deployment Effort |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face TEI** | Dedicated microservice, high throughput | **Yes (Token-level)** | **5x – 10x** | Low (Docker sidecar) |
| **Infinity (`infinity-emb`)** | Direct Python/FastAPI integration | **Yes (Async batching)** | **4x – 8x** | Very Low (Native library) |
| **ONNX Runtime (Optimum)** | Cross-platform (CPU/GPU) kernel fusion | Manual / Pipeline | **2x – 4x** | Low |
| **NVIDIA Triton + TensorRT** | Enterprise scale, lowest raw GPU latency | **Yes (Dynamic scheduler)**| **6x – 12x** | Medium / High |
| **`torch.compile`** | Zero architectural changes | No (batch=1 speedup) | **1.3x – 2x** | Trivial (1 line) |

---

### 1. Hugging Face TEI (Text Embeddings Inference)
* **Official Repository**: [https://github.com/huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
* **Official Docs**: [https://huggingface.co/docs/text-embeddings-inference](https://huggingface.co/docs/text-embeddings-inference)

#### Key Advantages
* **Written in Rust**: Bypasses the Python GIL completely; handles heavy concurrent network load with minimal latency.
* **Token-Level Dynamic Batching**: Groups sequences by token budget rather than fixed sequence counts, virtually eliminating padding overhead.
* **Optimized Attention Kernels**: Native support for FlashAttention-2 and Paged Attention adapted for encoders.
* **Quantization**: Built-in support for INT8, FP8, and FP16.

#### Deployment Example
Run TEI via Docker:
```bash
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id BAAI/bge-large-en-v1.5
```

FastAPI Gateway Client:
```python
import httpx
from fastapi import FastAPI

app = FastAPI()
client = httpx.AsyncClient(base_url="http://localhost:8080")

@app.post("/embed")
async def get_embeddings(texts: list[str]):
    response = await client.post("/embed", json={"inputs": texts})
    return {"embeddings": response.json()}
```

---

### 2. Infinity (`infinity-emb`)
* **Official Repository**: [https://github.com/michaelfeil/infinity](https://github.com/michaelfeil/infinity)

#### Key Advantages
* **Native FastAPI Integration**: Embeds directly into your existing FastAPI application via `AsyncEmbeddingEngine`.
* **Async Dynamic Batching**: Merges asynchronous concurrent HTTP requests on-the-fly before dispatching to the GPU.
* **Pluggable Backends**: Easily switch between PyTorch, `torch.compile`, ONNX Runtime, and TensorRT.
* **Multi-Task**: Works out of the box for text embeddings, cross-encoder rerankers, and image models (CLIP).

#### In-Process FastAPI Example
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

engine_args = EngineArgs(
    model_name_or_path="BAAI/bge-small-en-v1.5",
    engine="optimum",  # choices: "torch", "optimum" (ONNX), "tensorrt"
    batch_size=32
)
engine = AsyncEmbeddingEngine.from_args(engine_args)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine:
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/embed")
async def embed(text: str):
    # Concurrent calls are automatically batched together behind the scenes!
    embeddings, usage = await engine.embed(sentences=[text])
    return {"embedding": embeddings[0]}
```

---

### 3. ONNX Runtime via Hugging Face Optimum
* **Key Advantages**: Replaces PyTorch runtime with optimized C++ execution. Fuses LayerNorm, GeLU, and Attention into single kernels.

#### Export & Serve Example
```bash
pip install optimum[onnxruntime-gpu]
optimum-cli export onnx --model bert-base-uncased --task sequence-classification ./onnx_model/
```

```python
import asyncio
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")
model = ORTModelForSequenceClassification.from_pretrained(
    "./onnx_model", 
    provider="CUDAExecutionProvider" # Or "CPUExecutionProvider"
)

@app.post("/predict")
async def predict(text: str):
    def _run():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
        outputs = model(**inputs)
        return outputs.logits.argmax(dim=-1).item()

    return {"label": await asyncio.to_thread(_run)}
```

---

### 4. `torch.compile` (PyTorch 2.x Native)
* **Key Advantages**: 1-line change without converting model weights or changing deployment formats.

#### In-Process FastAPI Example
```python
import torch
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import asyncio

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to("cuda")
model.eval()

# 1-line compile optimization
compiled_model = torch.compile(model, mode="reduce-overhead")

# Warm-up pass during startup to trigger compilation once
with torch.inference_mode():
    dummy = torch.zeros((1, 128), dtype=torch.long, device="cuda")
    compiled_model(input_ids=dummy, attention_mask=dummy)

@app.post("/classify")
async def classify(text: str):
    def _infer():
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = compiled_model(**inputs)
        return outputs.logits.argmax(dim=-1).item()

    return {"label": await asyncio.to_thread(_infer)}
```

---

## Part 2: Deep Dive into `torch.compile`

### The Problem in Eager-Mode PyTorch
In standard eager-mode PyTorch:
```python
x = a + b
y = torch.relu(x)
z = y * c
```
Every operation launches a separate GPU kernel:
1. Python interpreter incurs dispatch overhead across PCIe.
2. The GPU reads `a` and `b` from High Bandwidth Memory (HBM/VRAM).
3. The GPU computes `a + b` in registers/SRAM and writes `x` back to HBM.
4. Next kernel reads `x` from HBM, applies `relu`, and writes `y` back to HBM.

Because elementwise activations and normalizations perform very few floating-point operations per byte transferred, the GPU is severely **memory-bandwidth bound**. Most time is spent waiting on memory transfer, not doing computation.

---

### The 3-Tier Architecture of `torch.compile`

```
               User Python Code
                      │
                      ▼
         [ 1. TorchDynamo (Frontend) ]
      • Bytecode analysis (PEP 523)
      • Guards & FX Graph extraction
      • Safe Graph Breaks
                      │
                      ▼
         [ 2. AOTAutograd (Middle) ]
      • Operator lowering to PrimTorch (~250 ops)
      • Ahead-Of-Time forward/backward graph tracing
                      │
                      ▼
         [ 3. TorchInductor (Backend) ]
      • Loop & kernel fusion
      • OpenAI Triton code generation
      • Compilation to hardware machine instructions
```

#### 1. TorchDynamo (Graph Capture via Bytecode Hook)
* Uses CPython's Frame Evaluation Hook (**PEP 523**).
* Inspects bytecode before execution, tracing tensor operations into an **FX Graph**.
* **Guards**: Tracks runtime assumptions (e.g., input shapes, tensor dtypes, global state). If guards hold, compiled code runs. If they fail, it re-compiles.
* **Graph Breaks**: If unsupported Python code or dynamic C-extensions execute, TorchDynamo gracefully falls back to the native Python interpreter for that section, then resumes compiled graph execution without crashing.
* **Debugging Graph Breaks**:
  ```python
  import torch

  def sample_fn(x):
      if x.sum() > 0:  # Data-dependent condition causes a graph break
          return x * 2
      return x * -1

  explanation = torch._dynamo.explain(sample_fn, torch.randn(10))
  print(f"Graph break count: {explanation.graph_break_count}")
  print(explanation.break_reasons)
  ```

#### 2. AOTAutograd & PrimTorch (Operator Decomposition)
* PyTorch has over 2,000 user-facing operations.
* **PrimTorch** lowers these into roughly **250 core primitive operations**.
* **AOTAutograd** captures forward and backward graphs before execution, facilitating optimal memory reuse.

#### 3. TorchInductor (Kernel Fusion & Triton Code Generation)
* Performs **loop fusion**: combines consecutive elementwise operations and reductions into a single kernel, keeping intermediate tensors inside high-speed GPU SRAM/registers.
* Emits **OpenAI Triton** code rather than raw CUDA C++, making compilation fast and hardware-adaptable.
* You can inspect the generated Triton code directly:
  ```bash
  TORCH_LOGS="output_code" python your_script.py
  ```

#### What Does `mode="reduce-overhead"` Do?
It integrates **CUDA Graphs**. Instead of launching kernels one-by-one from the CPU over PCIe, the entire execution graph is recorded directly onto the GPU driver. Playback is triggered by a single CPU instruction, eliminating CPU launch overhead.

---

## Part 3: Best Resources

1. **Foundational Theory**:
   * **[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)** by Horace He. *The essential primer on compute-bound vs memory-bandwidth-bound regimes and why operator fusion matters.*
2. **Architecture Paper**:
   * **[PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://arxiv.org/abs/2312.08188)** (Ansel et al., 2023).
3. **Official References**:
   * [PyTorch `torch.compiler` Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
   * [TorchDynamo Troubleshooting & Graph Break Guide](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
4. **Podcasts & Engineering Deep Dives**:
   * **[The PyTorch Dev Podcast](https://podcast.ezyang.com/)** by Edward Yang (Episodes on *TorchDynamo*, *Guards*, *Dynamic Shapes*, *Inductor*).
5. **Kernel Language**:
   * **[OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)**.

---

## Part 4: 4-Week Study Plan

### Week 1: GPU Hardware Fundamentals & The Roofline Model
* **Objective**: Understand memory bandwidth bottlenecks, arithmetic intensity, and SRAM vs HBM.
* **Read**: Horace He's *Making Deep Learning Go Brrrr From First Principles*.
* **Lab**: Profile an uncompiled model using `torch.profiler` and inspect kernel durations and memory copies.

### Week 2: TorchDynamo, Bytecode Interception & Graph Breaks
* **Objective**: Learn how Python frames are evaluated and how Dynamo creates FX graphs and guards.
* **Read**: PyTorch 2 Paper (Section 3).
* **Lab**: Use `torch._dynamo.explain` to identify and fix graph breaks in custom model modules.

### Week 3: PrimTorch Decomposition & TorchInductor
* **Objective**: Understand operator lowering and how loop fusion reduces memory round-trips.
* **Read**: PyTorch 2 Paper (Sections 4 & 5).
* **Lab**: Inspect generated Triton kernels by running with `TORCH_LOGS="output_code" python script.py`.

### Week 4: CUDA Graphs, Dynamic Shapes & Production Inference
* **Objective**: Master production deployment configurations.
* **Read**: OpenAI Triton Tutorial 01 (Vector Addition) and PyTorch CUDA Graphs guide.
* **Lab**: Benchmark inference throughput and P50/P99 latency with `torch.compile(mode="reduce-overhead")` vs eager mode under varying batch sizes.
