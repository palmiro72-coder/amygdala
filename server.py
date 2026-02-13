"""
AMYGDALA Server - FastAPI Web Interface
========================================
Patent Pending - Dr. Lucas do Prado Palmiro

Exposes the AMYGDALA system via REST API + WebSocket
for real-time neural visualization.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import AMYGDALA core
from amygdala_core import AmygdalaSystem, AmygdalaConfig, TaskComplexity

# =============================================================================
# CONFIGURATION
# =============================================================================

class ProcessRequest(BaseModel):
    """Request model for processing"""
    text: str = Field(..., min_length=1, max_length=10000)
    system_prompt: Optional[str] = None
    force_route: Optional[str] = None  # "limbic", "cortical", "prefrontal"
    metadata: Optional[Dict[str, Any]] = None

class AnalyzeRequest(BaseModel):
    """Request model for analysis only (no LLM call)"""
    text: str = Field(..., min_length=1, max_length=10000)

class ProcessResponse(BaseModel):
    """Response model"""
    success: bool
    request_id: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None
    routing: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    neural_state: Optional[Dict[str, Any]] = None

# =============================================================================
# LIFESPAN & GLOBALS
# =============================================================================

amygdala_system: Optional[AmygdalaSystem] = None
active_connections: list[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AMYGDALA on startup"""
    global amygdala_system
    
    print("\n🧠 Initializing AMYGDALA System...")
    
    # Check API keys
    hyperbolic_key = os.getenv("HYPERBOLIC_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    if not hyperbolic_key:
        print("⚠️  HYPERBOLIC_API_KEY not set!")
    else:
        print(f"✅ Hyperbolic API key loaded ({hyperbolic_key[:8]}...)")
    
    if not anthropic_key:
        print("⚠️  ANTHROPIC_API_KEY not set!")
    else:
        print(f"✅ Anthropic API key loaded ({anthropic_key[:8]}...)")
    
    config = AmygdalaConfig(
        hyperbolic_api_key=hyperbolic_key,
        claude_api_key=anthropic_key
    )
    
    amygdala_system = AmygdalaSystem(config)
    print("✅ AMYGDALA System ready!\n")
    
    yield
    
    print("\n👋 AMYGDALA System shutting down...")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="AMYGDALA Neural System",
    description="Neuromorphic AI Orchestration Layer - Patent Pending",
    version="1.0.0-alpha",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# NEURAL STATE COMPUTATION
# =============================================================================

def compute_neural_state(text: str, complexity: float, signal: str, route: str) -> Dict[str, Any]:
    """
    Compute neural state for visualization.
    Maps internal metrics to visual parameters.
    """
    text_lower = text.lower()
    
    # Neurotransmitter detection
    neurotransmitter = "SEROTONINA"
    neurotransmitter_level = 0.5
    state = "PROCESSAMENTO PADRÃO"
    
    urgent_keywords = ["urgente", "urgência", "emergência", "perigo", "crítico", "imediato", "socorro", "urgent", "emergency"]
    priority_keywords = ["importante", "prioridade", "crucial", "essencial", "atenção", "important", "priority"]
    
    if any(k in text_lower for k in urgent_keywords):
        neurotransmitter = "NOREPINEFRINA"
        neurotransmitter_level = 0.9
        state = "ESTADO DE ALERTA"
    elif any(k in text_lower for k in priority_keywords):
        neurotransmitter = "DOPAMINA"
        neurotransmitter_level = 0.7
        state = "FOCO AUMENTADO"
    elif complexity < 0.3:
        neurotransmitter = "SEROTONINA"
        neurotransmitter_level = 0.3
        state = "PROCESSAMENTO PADRÃO"
    
    # Amygdala activation based on emotional content
    emotional_keywords = ["medo", "raiva", "tristeza", "alegria", "ansiedade", "stress", "perigo", "ameaça"]
    amygdala_activation = sum(1 for k in emotional_keywords if k in text_lower) * 0.15
    amygdala_activation = min(amygdala_activation + complexity * 0.3, 1.0)
    
    # Stress level
    stress_level = "baixo"
    if amygdala_activation > 0.7:
        stress_level = "intenso"
    elif amygdala_activation > 0.4:
        stress_level = "elevado"
    
    # Keywords extraction
    all_keywords = urgent_keywords + priority_keywords + ["diagnóstico", "tratamento", "paciente", "análise"]
    found_keywords = [k.upper() for k in all_keywords if k in text_lower][:5]
    if not found_keywords:
        found_keywords = ["NEUTRO", "PADRÃO"]
    
    return {
        "neurotransmitter": neurotransmitter,
        "neurotransmitter_level": neurotransmitter_level,
        "state": state,
        "amygdala_activation": round(amygdala_activation, 2),
        "stress_level": stress_level,
        "keywords": found_keywords,
        "route_color": {
            "limbic": "#3b82f6",     # Blue
            "cortical": "#8b5cf6",    # Purple
            "prefrontal": "#ef4444"   # Red
        }.get(route, "#6b7280")
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve the neural interface"""
    html_path = os.path.join(os.path.dirname(__file__), "AMYGDALA_FINAL.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>AMYGDALA System</h1><p>Interface not found. Use /api/process endpoint.</p>")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "system": "AMYGDALA",
        "version": "1.0.0-alpha",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if not amygdala_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    return amygdala_system.get_stats()

@app.post("/api/analyze", response_model=Dict[str, Any])
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text complexity without calling LLM.
    Returns routing decision and neural state.
    """
    if not amygdala_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    text = request.text
    
    # Analyze complexity
    complexity = amygdala_system.router.analyze_complexity(text)
    
    # Determine signal
    signal = amygdala_system._determine_signal(text)
    
    # Determine route
    route = amygdala_system.router.determine_route(complexity, signal)
    
    # Compute neural state for visualization
    neural_state = compute_neural_state(text, complexity, signal.value, route.value)
    
    return {
        "success": True,
        "analysis": {
            "complexity_score": round(complexity, 3),
            "signal": signal.value,
            "route": route.value,
            "model_selection": {
                "limbic": "Llama 70B (Hyperbolic)",
                "cortical": "DeepSeek V3 (Hyperbolic)",
                "prefrontal": "Claude Sonnet (Anthropic)"
            }.get(route.value)
        },
        "neural_state": neural_state
    }

@app.post("/api/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    """
    Process text through the full AMYGDALA pipeline.
    Routes to appropriate LLM based on complexity.
    """
    if not amygdala_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Parse force_route if provided
    force_route = None
    if request.force_route:
        route_map = {
            "limbic": TaskComplexity.LIMBIC,
            "cortical": TaskComplexity.CORTICAL,
            "prefrontal": TaskComplexity.PREFRONTAL
        }
        force_route = route_map.get(request.force_route.lower())
    
    # Process through AMYGDALA
    result = await amygdala_system.process(
        prompt=request.text,
        system=request.system_prompt,
        force_route=force_route,
        metadata=request.metadata
    )
    
    # Add neural state for visualization
    neural_state = compute_neural_state(
        request.text,
        result["routing"]["complexity_score"],
        result["routing"]["signal"],
        result["routing"]["route"]
    )
    
    # Broadcast to WebSocket clients
    await broadcast_neural_update({
        "type": "process_complete",
        "neural_state": neural_state,
        "routing": result["routing"],
        "metrics": result["metrics"]
    })
    
    return ProcessResponse(
        success=result["success"],
        request_id=result.get("request_id"),
        content=result.get("content"),
        error=result.get("error"),
        routing=result.get("routing"),
        metrics=result.get("metrics"),
        neural_state=neural_state
    )

# =============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =============================================================================

async def broadcast_neural_update(data: Dict[str, Any]):
    """Broadcast neural state to all connected clients"""
    if not active_connections:
        return
    
    message = json.dumps(data)
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        active_connections.remove(conn)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time neural visualization"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "message": "AMYGDALA WebSocket connected",
            "stats": amygdala_system.get_stats() if amygdala_system else {}
        })
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "analyze":
                # Quick analysis without LLM
                text = message.get("text", "")
                if text and amygdala_system:
                    complexity = amygdala_system.router.analyze_complexity(text)
                    signal = amygdala_system._determine_signal(text)
                    route = amygdala_system.router.determine_route(complexity, signal)
                    neural_state = compute_neural_state(text, complexity, signal.value, route.value)
                    
                    await websocket.send_json({
                        "type": "analysis",
                        "complexity": complexity,
                        "signal": signal.value,
                        "route": route.value,
                        "neural_state": neural_state
                    })
            
            elif message.get("type") == "process":
                # Full processing
                text = message.get("text", "")
                if text and amygdala_system:
                    result = await amygdala_system.process(text)
                    neural_state = compute_neural_state(
                        text,
                        result["routing"]["complexity_score"],
                        result["routing"]["signal"],
                        result["routing"]["route"]
                    )
                    
                    await websocket.send_json({
                        "type": "result",
                        "success": result["success"],
                        "content": result.get("content"),
                        "routing": result["routing"],
                        "metrics": result["metrics"],
                        "neural_state": neural_state
                    })
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("AMYGDALA_PORT", "8000"))
    host = os.getenv("AMYGDALA_HOST", "0.0.0.0")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗ ███╗   ███╗██╗   ██╗ ██████╗ ██████╗  █████╗ ██╗      █████╗     ║
║    ██╔══██╗████╗ ████║╚██╗ ██╔╝██╔════╝ ██╔══██╗██╔══██╗██║     ██╔══██╗    ║
║    ███████║██╔████╔██║ ╚████╔╝ ██║  ███╗██║  ██║███████║██║     ███████║    ║
║    ██╔══██║██║╚██╔╝██║  ╚██╔╝  ██║   ██║██║  ██║██╔══██║██║     ██╔══██║    ║
║    ██║  ██║██║ ╚═╝ ██║   ██║   ╚██████╔╝██████╔╝██║  ██║███████╗██║  ██║    ║
║    ╚═╝  ╚═╝╚═╝     ╚═╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ║
║                                                                              ║
║                         WEB SERVER v1.0.0-alpha                              ║
║                    Patent Pending - Dr. Lucas Palmiro                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🌐 Starting server at http://{host}:{port}")
    print(f"📡 WebSocket at ws://{host}:{port}/ws")
    print(f"📊 API docs at http://{host}:{port}/docs\n")
    
    uvicorn.run(app, host=host, port=port)
