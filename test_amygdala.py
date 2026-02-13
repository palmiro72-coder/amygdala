#!/usr/bin/env python3
"""
AMYGDALA System - Professional Test Suite
==========================================
Tests the neuromorphic routing and inference capabilities.

Usage:
    python test_amygdala.py                    # Run all tests
    python test_amygdala.py --live             # Run with live API calls
    python test_amygdala.py --benchmark        # Run performance benchmark
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amygdala_core import (
    AmygdalaSystem, AmygdalaConfig, TaskComplexity,
    NeurotransmitterSignal, ThalamusRouter, HippocampusMemory
)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"  {status} {name}")
    if details:
        print(f"         {Colors.CYAN}{details}{Colors.ENDC}")


def test_complexity_analysis():
    """Test the complexity analysis algorithm"""
    print_header("COMPLEXITY ANALYSIS TESTS")
    
    config = AmygdalaConfig()
    router = ThalamusRouter(config)
    
    test_cases = [
        # (prompt, expected_min, expected_max, description)
        ("Hello", 0.0, 0.2, "Simple greeting"),
        ("What is 2+2?", 0.0, 0.2, "Simple math"),
        ("Explain quantum entanglement", 0.1, 0.4, "Educational explanation"),
        ("Analyze the patient's symptoms and provide a differential diagnosis for fatigue, weight loss, and polyuria", 0.6, 1.0, "Medical diagnosis"),
        ("Compare and evaluate the pros and cons of different machine learning architectures for time series prediction", 0.6, 1.0, "Complex technical analysis"),
        ("Write a treatment plan for type 2 diabetes with comorbid hypertension", 0.4, 0.8, "Medical treatment plan"),
    ]
    
    all_passed = True
    for prompt, expected_min, expected_max, description in test_cases:
        complexity = router.analyze_complexity(prompt)
        passed = expected_min <= complexity <= expected_max
        all_passed = all_passed and passed
        
        print_test(
            description,
            passed,
            f"Complexity: {complexity:.2f} (expected {expected_min:.1f}-{expected_max:.1f})"
        )
    
    return all_passed


def test_routing_logic():
    """Test the routing decision logic"""
    print_header("ROUTING LOGIC TESTS")
    
    config = AmygdalaConfig()
    router = ThalamusRouter(config)
    
    test_cases = [
        # (complexity, signal, expected_route)
        (0.1, NeurotransmitterSignal.SEROTONIN, TaskComplexity.LIMBIC),
        (0.5, NeurotransmitterSignal.SEROTONIN, TaskComplexity.CORTICAL),
        (0.9, NeurotransmitterSignal.SEROTONIN, TaskComplexity.PREFRONTAL),
        (0.5, NeurotransmitterSignal.NOREPINEPHRINE, TaskComplexity.PREFRONTAL),  # Urgency boosts
        (0.4, NeurotransmitterSignal.GABA, TaskComplexity.LIMBIC),  # Inhibition reduces (0.4 - 0.2 = 0.2)
        (0.2, NeurotransmitterSignal.DOPAMINE, TaskComplexity.CORTICAL),  # Priority boosts
    ]
    
    all_passed = True
    for complexity, signal, expected_route in test_cases:
        route = router.determine_route(complexity, signal)
        passed = route == expected_route
        all_passed = all_passed and passed
        
        print_test(
            f"Complexity {complexity:.1f} + {signal.value}",
            passed,
            f"Route: {route.value} (expected {expected_route.value})"
        )
    
    return all_passed


def test_memory_system():
    """Test the hippocampal memory system"""
    print_header("MEMORY SYSTEM TESTS")
    
    config = AmygdalaConfig()
    memory = HippocampusMemory(config)
    
    all_passed = True
    
    # Test 1: Store and retrieve
    hash1 = memory.store("Patient has diabetes", emotional_score=0.8)
    hash2 = memory.store("Blood glucose is elevated", emotional_score=0.5)
    hash3 = memory.store("Weather is sunny today", emotional_score=0.1)
    
    retrieved = memory.retrieve("glucose levels", top_k=2)
    
    passed = len(retrieved) >= 2
    all_passed = all_passed and passed
    print_test("Store and retrieve", passed, f"Retrieved {len(retrieved)} memories")
    
    # Test 2: Emotional prioritization
    scores = [m.emotional_score for m in retrieved]
    passed = scores == sorted(scores, key=abs, reverse=True)
    all_passed = all_passed and passed
    print_test("Emotional prioritization", passed, f"Scores: {scores}")
    
    # Test 3: Working memory consolidation
    for i in range(15):
        memory.store(f"Test memory {i}", emotional_score=0.1 * i)
    
    passed = len(memory.working_memory) <= 10
    all_passed = all_passed and passed
    print_test("Working memory limit", passed, f"Working: {len(memory.working_memory)}, Long-term: {len(memory.long_term_memory)}")
    
    # Test 4: Emotional decay
    initial_score = list(memory.emotional_index.values())[0] if memory.emotional_index else 0
    memory.decay_emotions()
    decayed_score = list(memory.emotional_index.values())[0] if memory.emotional_index else 0
    
    passed = decayed_score < initial_score if initial_score > 0 else True
    all_passed = all_passed and passed
    print_test("Emotional decay", passed, f"Initial: {initial_score:.3f}, Decayed: {decayed_score:.3f}")
    
    return all_passed


async def test_live_inference():
    """Test live inference with actual API calls"""
    print_header("LIVE INFERENCE TESTS")
    
    # Check API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_hyperbolic = bool(os.getenv("HYPERBOLIC_API_KEY"))
    
    if not has_anthropic and not has_hyperbolic:
        print(f"  {Colors.YELLOW}⚠ Skipped: No API keys configured{Colors.ENDC}")
        return True
    
    config = AmygdalaConfig()
    amygdala = AmygdalaSystem(config)
    
    all_passed = True
    
    test_prompts = [
        ("Hello, how are you?", TaskComplexity.LIMBIC, "Simple greeting"),
        ("Explain how photosynthesis works", TaskComplexity.CORTICAL, "Educational explanation"),
    ]
    
    # Only add complex test if Claude is available
    if has_anthropic:
        test_prompts.append((
            "Analyze the pathophysiology of diabetic ketoacidosis and provide a treatment algorithm",
            TaskComplexity.PREFRONTAL,
            "Complex medical reasoning"
        ))
    
    for prompt, expected_route, description in test_prompts:
        try:
            result = await amygdala.process(prompt)
            
            passed = result["success"]
            route_match = result["routing"]["route"] == expected_route.value
            
            all_passed = all_passed and passed
            
            print_test(
                description,
                passed,
                f"Route: {result['routing']['route']} | "
                f"Backend: {result['routing']['backend']} | "
                f"Latency: {result['metrics'].get('latency_ms', 0):.0f}ms"
            )
            
            if passed:
                preview = result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
                print(f"         {Colors.CYAN}Response: {preview}{Colors.ENDC}")
                
        except Exception as e:
            print_test(description, False, f"Error: {str(e)}")
            all_passed = False
    
    return all_passed


async def benchmark():
    """Run performance benchmark"""
    print_header("PERFORMANCE BENCHMARK")
    
    if not os.getenv("HYPERBOLIC_API_KEY"):
        print(f"  {Colors.YELLOW}⚠ Skipped: HYPERBOLIC_API_KEY not set{Colors.ENDC}")
        return
    
    config = AmygdalaConfig()
    amygdala = AmygdalaSystem(config)
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly",
        "How does gradient descent work?",
        "What are transformers in AI?",
        "Describe backpropagation",
    ]
    
    print(f"  Running {len(prompts)} requests through AMYGDALA...\n")
    
    latencies = []
    start_total = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        start = time.time()
        result = await amygdala.process(prompt, force_route=TaskComplexity.LIMBIC)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if result["success"] else f"{Colors.RED}✗{Colors.ENDC}"
        print(f"  {status} Request {i}: {latency:.0f}ms | {result['routing']['backend']}")
    
    total_time = time.time() - start_total
    
    print(f"\n  {Colors.BOLD}Results:{Colors.ENDC}")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Avg latency: {sum(latencies)/len(latencies):.0f}ms")
    print(f"    Min latency: {min(latencies):.0f}ms")
    print(f"    Max latency: {max(latencies):.0f}ms")
    print(f"    Throughput: {len(prompts)/total_time:.2f} req/s")
    
    # Print system stats
    stats = amygdala.get_stats()
    print(f"\n  {Colors.BOLD}System Stats:{Colors.ENDC}")
    print(f"    Route distribution: {json.dumps(stats['route_distribution'], indent=6)}")


async def main():
    """Main test runner"""
    print(f"""
{Colors.CYAN}{Colors.BOLD}
    █████╗ ███╗   ███╗██╗   ██╗ ██████╗ ██████╗  █████╗ ██╗      █████╗ 
   ██╔══██╗████╗ ████║╚██╗ ██╔╝██╔════╝ ██╔══██╗██╔══██╗██║     ██╔══██╗
   ███████║██╔████╔██║ ╚████╔╝ ██║  ███╗██║  ██║███████║██║     ███████║
   ██╔══██║██║╚██╔╝██║  ╚██╔╝  ██║   ██║██║  ██║██╔══██║██║     ██╔══██║
   ██║  ██║██║ ╚═╝ ██║   ██║   ╚██████╔╝██████╔╝██║  ██║███████╗██║  ██║
   ╚═╝  ╚═╝╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
{Colors.ENDC}
                    {Colors.BOLD}Professional Test Suite{Colors.ENDC}
                    Patent Pending - Dr. Lucas Palmiro
                    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
    
    args = sys.argv[1:]
    
    results = []
    
    # Unit tests (always run)
    if "--benchmark" not in args:
        results.append(("Complexity Analysis", test_complexity_analysis()))
        results.append(("Routing Logic", test_routing_logic()))
        results.append(("Memory System", test_memory_system()))
    
    # Live tests (only with --live flag)
    if "--live" in args:
        results.append(("Live Inference", await test_live_inference()))
    
    # Benchmark (only with --benchmark flag)
    if "--benchmark" in args:
        await benchmark()
        return
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    for name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.RED}FAIL{Colors.ENDC}"
        print(f"  [{status}] {name}")
    
    print(f"\n  {Colors.BOLD}Total: {total_passed}/{total_tests} test suites passed{Colors.ENDC}")
    
    if total_passed == total_tests:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}\n")
    else:
        print(f"\n  {Colors.RED}{Colors.BOLD}⚠ Some tests failed{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
