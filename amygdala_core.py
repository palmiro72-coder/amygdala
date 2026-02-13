"""
AMYGDALA System - Neuromorphic AI Orchestration Layer
======================================================
Patent Pending - Dr. Lucas do Prado Palmiro
Version: 1.0.0-alpha

Architecture inspired by the human limbic system:
- Amygdala: Emotional scoring and priority routing
- Hippocampus: Semantic memory and context retrieval
- Thalamus: Request routing and load balancing
- Cortex: High-level reasoning (Claude API)

This MVP implements the core routing logic using:
- Hyperbolic API for fast, cheap inference (Llama, DeepSeek)
- Claude API for complex reasoning tasks
"""

import os
import json
import time
import hashlib
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import httpx
from openai import OpenAI, AsyncOpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AmygdalaConfig:
    """System configuration with sensible defaults"""
    
    # API Keys (from environment)
    claude_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    hyperbolic_api_key: str = field(default_factory=lambda: os.getenv("HYPERBOLIC_API_KEY", ""))
    
    # Model selection
    claude_model: str = "claude-sonnet-4-20250514"
    hyperbolic_model_fast: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    hyperbolic_model_reasoning: str = "deepseek-ai/DeepSeek-V3"
    
    # Routing thresholds
    complexity_threshold_low: float = 0.3   # Below: use fast model
    complexity_threshold_high: float = 0.7  # Above: use Claude
    
    # Performance settings
    max_tokens_fast: int = 1024
    max_tokens_complex: int = 4096
    timeout_seconds: int = 60
    
    # Memory settings
    memory_cache_size: int = 1000
    emotional_decay_rate: float = 0.95


class TaskComplexity(Enum):
    """Task complexity levels mapped to neural regions"""
    REFLEX = "reflex"           # Instant, cached responses
    LIMBIC = "limbic"           # Emotional/simple tasks -> Hyperbolic fast
    CORTICAL = "cortical"       # Reasoning tasks -> Hyperbolic reasoning
    PREFRONTAL = "prefrontal"   # Complex reasoning -> Claude


class NeurotransmitterSignal(Enum):
    """Digital neurotransmitter signals for routing"""
    DOPAMINE = "DOP"    # Reward/priority boost
    NOREPINEPHRINE = "NOR"  # Urgency/attention
    SEROTONIN = "SER"   # Calm/standard processing
    GABA = "GABA"       # Inhibition/rate limiting
    GLUTAMATE = "GLU"   # Excitation/parallel processing


# =============================================================================
# MEMORY SUBSYSTEM (Hippocampus)
# =============================================================================

@dataclass
class MemoryTrace:
    """A single memory trace with emotional valence"""
    content: str
    embedding_hash: str
    emotional_score: float  # -1.0 to 1.0
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HippocampusMemory:
    """
    Semantic memory system inspired by hippocampal function.
    Implements:
    - Working memory (short-term context)
    - Long-term memory (consolidated knowledge)
    - Emotional tagging for priority retrieval
    """
    
    def __init__(self, config: AmygdalaConfig):
        self.config = config
        self.working_memory: List[MemoryTrace] = []
        self.long_term_memory: Dict[str, MemoryTrace] = {}
        self.emotional_index: Dict[str, float] = {}
        
    def _compute_hash(self, content: str) -> str:
        """Compute semantic hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def store(self, content: str, emotional_score: float = 0.0, 
              metadata: Optional[Dict] = None) -> str:
        """Store content with emotional tagging"""
        trace_hash = self._compute_hash(content)
        
        trace = MemoryTrace(
            content=content,
            embedding_hash=trace_hash,
            emotional_score=emotional_score,
            metadata=metadata or {}
        )
        
        # Add to working memory
        self.working_memory.append(trace)
        if len(self.working_memory) > 10:
            # Consolidate oldest to long-term
            old_trace = self.working_memory.pop(0)
            self.long_term_memory[old_trace.embedding_hash] = old_trace
        
        # Update emotional index
        self.emotional_index[trace_hash] = emotional_score
        
        return trace_hash
    
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryTrace]:
        """Retrieve relevant memories, prioritized by emotional score"""
        query_hash = self._compute_hash(query)
        
        # Combine working and long-term memory
        all_traces = list(self.working_memory) + list(self.long_term_memory.values())
        
        # Sort by emotional score (higher = more relevant for urgent queries)
        sorted_traces = sorted(
            all_traces, 
            key=lambda t: abs(t.emotional_score),
            reverse=True
        )
        
        return sorted_traces[:top_k]
    
    def decay_emotions(self):
        """Apply emotional decay (forgetting curve)"""
        for trace_hash in self.emotional_index:
            self.emotional_index[trace_hash] *= self.config.emotional_decay_rate


# =============================================================================
# ROUTING SUBSYSTEM (Thalamus)
# =============================================================================

class ThalamusRouter:
    """
    Request routing system inspired by thalamic relay function.
    Routes requests to appropriate processing backends based on:
    - Complexity analysis
    - Emotional urgency
    - Resource availability
    """
    
    def __init__(self, config: AmygdalaConfig):
        self.config = config
        self.request_count = 0
        self.latency_history: Dict[str, List[float]] = {
            "hyperbolic_fast": [],
            "hyperbolic_reasoning": [],
            "claude": []
        }
    
    def analyze_complexity(self, prompt: str, context: Optional[str] = None) -> float:
        """
        Analyze task complexity (0.0 to 1.0)
        Uses heuristics inspired by cognitive load theory
        """
        score = 0.0
        
        # Length factor (more granular)
        word_count = len(prompt.split())
        if word_count > 20:
            score += 0.1
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.1
        if word_count > 200:
            score += 0.1
            
        prompt_lower = prompt.lower()
        
        # High complexity indicators (strong signals)
        high_complexity_keywords = [
            "analyze", "evaluate", "synthesize", "compare and", 
            "differential diagnosis", "treatment plan", "treatment algorithm",
            "pathophysiology", "pros and cons", "architecture",
            "algorithm", "methodology", "statistical analysis"
        ]
        high_matches = sum(1 for kw in high_complexity_keywords if kw in prompt_lower)
        score += min(high_matches * 0.25, 0.5)
        
        # Medium complexity indicators
        medium_complexity_keywords = [
            "explain", "describe", "how does", "what is the",
            "compare", "contrast", "create", "design", "implement",
            "diagnose", "prognosis", "research", "hypothesis"
        ]
        medium_matches = sum(1 for kw in medium_complexity_keywords if kw in prompt_lower)
        score += min(medium_matches * 0.15, 0.3)
        
        # Medical/clinical indicators (domain-specific boost for safety)
        medical_keywords = [
            "patient", "symptom", "diagnosis", "medication", "dosage",
            "contraindication", "etiology", "clinical", "therapy",
            "diabetes", "hypertension", "treatment", "comorbid"
        ]
        medical_matches = sum(1 for mk in medical_keywords if mk in prompt_lower)
        if medical_matches >= 2:
            score += 0.3  # Strong medical context
        elif medical_matches >= 1:
            score += 0.15  # Some medical context
        
        # Technical indicators
        technical_keywords = [
            "code", "programming", "machine learning", "neural",
            "api", "database", "system", "infrastructure"
        ]
        if any(tk in prompt_lower for tk in technical_keywords):
            score += 0.1
        
        # Question complexity boosters
        complex_question_patterns = [
            "why", "how would", "what if", "what are the implications",
            "trade-offs", "advantages and disadvantages"
        ]
        if any(qp in prompt_lower for qp in complex_question_patterns):
            score += 0.1
        
        return min(score, 1.0)
    
    def determine_route(self, complexity: float, 
                        signal: NeurotransmitterSignal = NeurotransmitterSignal.SEROTONIN
                        ) -> TaskComplexity:
        """Determine processing route based on complexity and signal"""
        
        # Dopamine boost increases routing to higher-level processing
        complexity_adjusted = complexity
        if signal == NeurotransmitterSignal.DOPAMINE:
            complexity_adjusted += 0.2
        elif signal == NeurotransmitterSignal.NOREPINEPHRINE:
            complexity_adjusted += 0.3  # Urgency -> higher processing
        elif signal == NeurotransmitterSignal.GABA:
            complexity_adjusted -= 0.2  # Inhibition -> lower processing
        
        complexity_adjusted = max(0.0, min(1.0, complexity_adjusted))
        
        if complexity_adjusted < self.config.complexity_threshold_low:
            return TaskComplexity.LIMBIC
        elif complexity_adjusted < self.config.complexity_threshold_high:
            return TaskComplexity.CORTICAL
        else:
            return TaskComplexity.PREFRONTAL
    
    def record_latency(self, backend: str, latency_ms: float):
        """Record latency for adaptive routing"""
        if backend in self.latency_history:
            self.latency_history[backend].append(latency_ms)
            # Keep only last 100 measurements
            if len(self.latency_history[backend]) > 100:
                self.latency_history[backend] = self.latency_history[backend][-100:]
    
    def get_avg_latency(self, backend: str) -> float:
        """Get average latency for backend"""
        history = self.latency_history.get(backend, [])
        return sum(history) / len(history) if history else 0.0


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================

class InferenceBackend(ABC):
    """Abstract base for inference backends"""
    
    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        pass


class HyperbolicBackend(InferenceBackend):
    """Hyperbolic API backend for fast/cheap inference"""
    
    def __init__(self, config: AmygdalaConfig, model_type: Literal["fast", "reasoning"] = "fast"):
        self.config = config
        self.model = (config.hyperbolic_model_fast if model_type == "fast" 
                      else config.hyperbolic_model_reasoning)
        self.model_type = model_type
        self.client = AsyncOpenAI(
            api_key=config.hyperbolic_api_key,
            base_url="https://api.hyperbolic.xyz/v1"
        )
    
    async def generate(self, prompt: str, system: Optional[str] = None,
                       max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate completion via Hyperbolic"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": self.model,
                "backend": f"hyperbolic_{self.model_type}",
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "backend": f"hyperbolic_{self.model_type}"
            }


class ClaudeBackend(InferenceBackend):
    """Claude API backend for complex reasoning"""
    
    def __init__(self, config: AmygdalaConfig):
        self.config = config
        self.client = None  # Initialized lazily
    
    async def generate(self, prompt: str, system: Optional[str] = None,
                       max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate completion via Claude API"""
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                headers = {
                    "x-api-key": self.config.claude_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                
                payload = {
                    "model": self.config.claude_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                if system:
                    payload["system"] = system
                
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "content": data["content"][0]["text"],
                    "model": self.config.claude_model,
                    "backend": "claude",
                    "latency_ms": latency_ms,
                    "usage": {
                        "prompt_tokens": data["usage"]["input_tokens"],
                        "completion_tokens": data["usage"]["output_tokens"]
                    }
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "backend": "claude"
            }


# =============================================================================
# MAIN AMYGDALA ORCHESTRATOR
# =============================================================================

class AmygdalaSystem:
    """
    Main orchestration system - the "emotional brain" of the architecture.
    
    Coordinates:
    - Memory (Hippocampus)
    - Routing (Thalamus)  
    - Inference backends (Cortex layers)
    
    Implements neuromorphic decision-making for optimal inference routing.
    """
    
    def __init__(self, config: Optional[AmygdalaConfig] = None):
        self.config = config or AmygdalaConfig()
        
        # Initialize subsystems
        self.memory = HippocampusMemory(self.config)
        self.router = ThalamusRouter(self.config)
        
        # Initialize backends
        self.backends = {
            TaskComplexity.LIMBIC: HyperbolicBackend(self.config, "fast"),
            TaskComplexity.CORTICAL: HyperbolicBackend(self.config, "reasoning"),
            TaskComplexity.PREFRONTAL: ClaudeBackend(self.config)
        }
        
        # Metrics
        self.total_requests = 0
        self.route_distribution: Dict[str, int] = {
            "limbic": 0, "cortical": 0, "prefrontal": 0
        }
    
    def _determine_signal(self, prompt: str, metadata: Optional[Dict] = None) -> NeurotransmitterSignal:
        """Determine neurotransmitter signal based on context"""
        prompt_lower = prompt.lower()
        
        # Urgency indicators -> Norepinephrine
        urgent_keywords = ["urgent", "emergency", "asap", "critical", "immediately"]
        if any(uk in prompt_lower for uk in urgent_keywords):
            return NeurotransmitterSignal.NOREPINEPHRINE
        
        # Priority/reward indicators -> Dopamine
        priority_keywords = ["important", "priority", "crucial", "key", "essential"]
        if any(pk in prompt_lower for pk in priority_keywords):
            return NeurotransmitterSignal.DOPAMINE
        
        # Rate limiting indicators -> GABA
        if metadata and metadata.get("rate_limited"):
            return NeurotransmitterSignal.GABA
        
        return NeurotransmitterSignal.SEROTONIN
    
    async def process(self, prompt: str, 
                      system: Optional[str] = None,
                      context: Optional[str] = None,
                      force_route: Optional[TaskComplexity] = None,
                      metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a request through the AMYGDALA system.
        
        Args:
            prompt: The user's input
            system: Optional system prompt
            context: Optional context from memory
            force_route: Force a specific route (bypasses analysis)
            metadata: Additional request metadata
        
        Returns:
            Response dict with content, routing info, and metrics
        """
        self.total_requests += 1
        request_id = f"amg_{self.total_requests}_{int(time.time())}"
        
        # Analyze complexity
        complexity = self.router.analyze_complexity(prompt, context)
        
        # Determine neurotransmitter signal
        signal = self._determine_signal(prompt, metadata)
        
        # Determine route
        if force_route:
            route = force_route
        else:
            route = self.router.determine_route(complexity, signal)
        
        # Update metrics
        self.route_distribution[route.value] = self.route_distribution.get(route.value, 0) + 1
        
        # Get appropriate backend
        backend = self.backends[route]
        
        # Build enhanced prompt with context
        enhanced_prompt = prompt
        if context:
            enhanced_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
        
        # Retrieve relevant memories
        memories = self.memory.retrieve(prompt, top_k=3)
        if memories:
            memory_context = "\n".join([f"- {m.content[:200]}..." for m in memories])
            enhanced_prompt = f"Relevant history:\n{memory_context}\n\n{enhanced_prompt}"
        
        # Execute inference
        max_tokens = (self.config.max_tokens_fast if route in [TaskComplexity.LIMBIC, TaskComplexity.CORTICAL]
                      else self.config.max_tokens_complex)
        
        result = await backend.generate(enhanced_prompt, system, max_tokens)
        
        # Store in memory with emotional score based on success
        emotional_score = 0.5 if result["success"] else -0.5
        self.memory.store(prompt, emotional_score, {"response": result.get("content", "")[:500]})
        
        # Record latency
        if result["success"]:
            self.router.record_latency(result["backend"], result["latency_ms"])
        
        # Decay emotional memories
        self.memory.decay_emotions()
        
        return {
            "request_id": request_id,
            "success": result["success"],
            "content": result.get("content"),
            "error": result.get("error"),
            "routing": {
                "complexity_score": complexity,
                "signal": signal.value,
                "route": route.value,
                "backend": result["backend"],
                "model": result.get("model")
            },
            "metrics": {
                "latency_ms": result.get("latency_ms"),
                "usage": result.get("usage")
            },
            "system_stats": {
                "total_requests": self.total_requests,
                "route_distribution": self.route_distribution.copy()
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_requests": self.total_requests,
            "route_distribution": self.route_distribution,
            "avg_latencies": {
                backend: self.router.get_avg_latency(backend)
                for backend in self.router.latency_history.keys()
            },
            "memory_stats": {
                "working_memory_size": len(self.memory.working_memory),
                "long_term_memory_size": len(self.memory.long_term_memory)
            }
        }


# =============================================================================
# CLI INTERFACE FOR TESTING
# =============================================================================

async def main():
    """CLI interface for testing AMYGDALA"""
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗ ███╗   ███╗██╗   ██╗ ██████╗ ██████╗  █████╗ ██╗      █████╗     ║
║    ██╔══██╗████╗ ████║╚██╗ ██╔╝██╔════╝ ██╔══██╗██╔══██╗██║     ██╔══██╗    ║
║    ███████║██╔████╔██║ ╚████╔╝ ██║  ███╗██║  ██║███████║██║     ███████║    ║
║    ██╔══██║██║╚██╔╝██║  ╚██╔╝  ██║   ██║██║  ██║██╔══██║██║     ██╔══██║    ║
║    ██║  ██║██║ ╚═╝ ██║   ██║   ╚██████╔╝██████╔╝██║  ██║███████╗██║  ██║    ║
║    ╚═╝  ╚═╝╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ║
║                                                                              ║
║                    Neuromorphic AI Orchestration System                      ║
║                         Version 1.0.0-alpha                                  ║
║                    Patent Pending - Dr. Lucas Palmiro                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Claude backend will fail.")
    if not os.getenv("HYPERBOLIC_API_KEY"):
        print("⚠️  Warning: HYPERBOLIC_API_KEY not set. Hyperbolic backend will fail.")
    
    # Initialize system
    config = AmygdalaConfig()
    amygdala = AmygdalaSystem(config)
    
    print("\n🧠 AMYGDALA System initialized.")
    print("   - Backends: Hyperbolic (Llama 70B, DeepSeek V3), Claude (Sonnet)")
    print("   - Memory: Hippocampal storage active")
    print("   - Routing: Thalamic relay active")
    print("\nType 'quit' to exit, 'stats' for system stats, or enter a prompt.\n")
    
    while True:
        try:
            prompt = input("🔮 You: ").strip()
            
            if not prompt:
                continue
            if prompt.lower() == "quit":
                print("\n👋 AMYGDALA shutting down. Goodbye!")
                break
            if prompt.lower() == "stats":
                stats = amygdala.get_stats()
                print(f"\n📊 System Stats:")
                print(f"   Total requests: {stats['total_requests']}")
                print(f"   Route distribution: {stats['route_distribution']}")
                print(f"   Avg latencies: {stats['avg_latencies']}")
                print(f"   Memory: {stats['memory_stats']}\n")
                continue
            
            # Process request
            print("\n⚡ Processing...")
            result = await amygdala.process(prompt)
            
            if result["success"]:
                print(f"\n🧠 AMYGDALA [{result['routing']['route'].upper()}]: {result['content']}")
                print(f"\n   📍 Route: {result['routing']['route']} | "
                      f"Backend: {result['routing']['backend']} | "
                      f"Complexity: {result['routing']['complexity_score']:.2f} | "
                      f"Latency: {result['metrics']['latency_ms']:.0f}ms\n")
            else:
                print(f"\n❌ Error: {result['error']}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 AMYGDALA interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
