# 🧠 AMYGDALA System

> **Neuromorphic AI Orchestration Layer**  
> Patent Pending - Dr. Lucas do Prado Palmiro  
> Version 1.0.0-alpha

---

## Overview

AMYGDALA (Advanced Multi-modal Yet Generalized Dynamic Adaptive Learning Architecture) is a neuromorphic orchestration system that intelligently routes AI inference requests between multiple backends based on task complexity, urgency, and resource availability.

Inspired by the human limbic system, AMYGDALA implements:

| Component | Biological Analog | Function |
|-----------|------------------|----------|
| **Router** | Thalamus | Request routing and load balancing |
| **Memory** | Hippocampus | Semantic memory and context retrieval |
| **Scoring** | Amygdala | Emotional/priority scoring for decisions |
| **Reasoning** | Cortex | High-level inference (Claude API) |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           AMYGDALA SYSTEM               │
                    │      Neuromorphic Orchestration         │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           THALAMUS ROUTER               │
                    │   Complexity Analysis & Route Decision  │
                    └───┬─────────────┬─────────────┬─────────┘
                        │             │             │
           ┌────────────▼──┐   ┌──────▼──────┐   ┌──▼────────────┐
           │    LIMBIC     │   │  CORTICAL   │   │  PREFRONTAL   │
           │  (Fast/Cheap) │   │ (Reasoning) │   │   (Complex)   │
           │               │   │             │   │               │
           │  Hyperbolic   │   │  Hyperbolic │   │    Claude     │
           │  Llama 70B    │   │  DeepSeek   │   │    Sonnet     │
           └───────────────┘   └─────────────┘   └───────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         HIPPOCAMPUS MEMORY              │
                    │   Working Memory + Long-term Storage    │
                    │   Emotional Scoring + Decay Curves      │
                    └─────────────────────────────────────────┘
```

---

## Features

### Intelligent Routing
- **Complexity Analysis**: Analyzes prompts using cognitive load heuristics
- **Medical Domain Boost**: Automatic complexity increase for medical queries (safety)
- **Neurotransmitter Signals**: Urgency (NOR), priority (DOP), standard (SER), rate-limiting (GABA)

### Multi-Backend Support
- **Hyperbolic Fast** (Llama 70B): Simple queries, low latency
- **Hyperbolic Reasoning** (DeepSeek V3): Moderate complexity
- **Claude** (Sonnet): Complex reasoning, medical decisions

### Memory System
- **Working Memory**: Last 10 interactions with fast access
- **Long-term Memory**: Consolidated storage with hash indexing
- **Emotional Decay**: Forgetting curve for memory prioritization

---

## Installation

```bash
# Clone or copy the amygdala directory
cd amygdala

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

Set environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export HYPERBOLIC_API_KEY="..."
```

Or use a `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key
HYPERBOLIC_API_KEY=your-hyperbolic-key
```

---

## Usage

### CLI Interface

```bash
python amygdala_core.py
```

Interactive prompt:
```
🔮 You: Explain diabetes treatment
⚡ Processing...

🧠 AMYGDALA [CORTICAL]: Diabetes treatment involves...

   📍 Route: cortical | Backend: hyperbolic_reasoning | Complexity: 0.45 | Latency: 892ms
```

### Programmatic Usage

```python
import asyncio
from amygdala_core import AmygdalaSystem, TaskComplexity

async def main():
    # Initialize system
    amygdala = AmygdalaSystem()
    
    # Simple query (routes to Hyperbolic fast)
    result = await amygdala.process("What is insulin?")
    print(result["content"])
    
    # Complex query (routes to Claude)
    result = await amygdala.process(
        "Analyze differential diagnosis for patient with polyuria, polydipsia, and weight loss"
    )
    print(result["content"])
    
    # Force specific route
    result = await amygdala.process(
        "Quick calculation: 15% of 200",
        force_route=TaskComplexity.LIMBIC
    )
    
    # Get system stats
    stats = amygdala.get_stats()
    print(stats)

asyncio.run(main())
```

---

## Testing

```bash
# Run unit tests
python test_amygdala.py

# Run with live API calls
python test_amygdala.py --live

# Run performance benchmark
python test_amygdala.py --benchmark
```

---

## Routing Logic

### Complexity Thresholds

| Score | Route | Backend | Use Case |
|-------|-------|---------|----------|
| 0.0 - 0.3 | LIMBIC | Hyperbolic Llama 70B | Simple queries, greetings |
| 0.3 - 0.7 | CORTICAL | Hyperbolic DeepSeek | Explanations, moderate reasoning |
| 0.7 - 1.0 | PREFRONTAL | Claude Sonnet | Complex analysis, medical decisions |

### Neurotransmitter Modifiers

| Signal | Effect | Trigger Keywords |
|--------|--------|-----------------|
| DOPAMINE | +0.2 complexity | important, priority, crucial |
| NOREPINEPHRINE | +0.3 complexity | urgent, emergency, critical |
| SEROTONIN | baseline | (default) |
| GABA | -0.2 complexity | (rate limiting) |

---

## Patent Portfolio

AMYGDALA is part of a 7-patent portfolio covering:

1. **Hierarchical Neuromorphic Memory System** (G06F 12/08)
2. **NCM Module** - NVMe-Class Memory (G11C 11/406)
3. **Harmonic Resonant Clock System** (H03B 5/32)
4. **Toroidal Thermal Flow Cabinet** (H05K 7/20)
5. **NeuroBus Communication Protocol** (H04L 49/00)
6. **Reflex Thermal Protection** (G06F 1/20)
7. **Geomagnetic PCB Routing** (H05K 1/02)

---

## Roadmap

- [ ] Vector embeddings for semantic memory
- [ ] Adaptive threshold learning
- [ ] Multi-model ensemble voting
- [ ] Hardware integration (FPGA thermal protection)
- [ ] Kubernetes deployment with auto-scaling

---

## License

Proprietary - Patent Pending  
© 2026 Dr. Lucas do Prado Palmiro

---

## Contact

**Dr. Lucas do Prado Palmiro**  
Endocrinology & Metabolism | AI Systems  
CREMESP 139089 | RQE 75065  
Staff Physician, Hospital Israelita Albert Einstein

Clínica Palmiros | Bella Derm  
Rua Borges Lagoa, 971 - Vila Clementino  
São Paulo, SP - Brazil
