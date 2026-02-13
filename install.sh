#!/bin/bash
# =============================================================================
# AMYGDALA System - Installation Script
# =============================================================================
# Patent Pending - Dr. Lucas do Prado Palmiro
# Run this script on your server to set up AMYGDALA
# =============================================================================

set -e

echo "
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AMYGDALA INSTALLER                                   ║
║                    Patent Pending - Dr. Lucas Palmiro                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Not running as root. Some operations may fail.${NC}"
fi

# Configuration
INSTALL_DIR="${AMYGDALA_DIR:-/opt/amygdala}"
AMYGDALA_USER="${AMYGDALA_USER:-amygdala}"
AMYGDALA_PORT="${AMYGDALA_PORT:-8000}"

echo "📁 Installation directory: $INSTALL_DIR"
echo "👤 Service user: $AMYGDALA_USER"
echo "🌐 Port: $AMYGDALA_PORT"
echo ""

# =============================================================================
# STEP 1: System Dependencies
# =============================================================================
echo -e "${GREEN}[1/6] Installing system dependencies...${NC}"

if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv git curl
elif command -v yum &> /dev/null; then
    yum install -y python3 python3-pip git curl
elif command -v pacman &> /dev/null; then
    pacman -S --noconfirm python python-pip git curl
else
    echo -e "${YELLOW}⚠️  Unknown package manager. Please install Python 3.10+ manually.${NC}"
fi

echo "✅ System dependencies installed"

# =============================================================================
# STEP 2: Create User & Directory
# =============================================================================
echo -e "${GREEN}[2/6] Setting up user and directories...${NC}"

# Create user if doesn't exist
if ! id "$AMYGDALA_USER" &>/dev/null; then
    useradd -r -s /bin/false "$AMYGDALA_USER" 2>/dev/null || true
fi

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "✅ Directory created: $INSTALL_DIR"

# =============================================================================
# STEP 3: Python Virtual Environment
# =============================================================================
echo -e "${GREEN}[3/6] Creating Python virtual environment...${NC}"

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

echo "✅ Virtual environment created"

# =============================================================================
# STEP 4: Install Python Dependencies
# =============================================================================
echo -e "${GREEN}[4/6] Installing Python dependencies...${NC}"

pip install -q \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    httpx>=0.25.0 \
    openai>=1.0.0 \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0 \
    websockets>=12.0

echo "✅ Python dependencies installed"

# =============================================================================
# STEP 5: Create Environment File
# =============================================================================
echo -e "${GREEN}[5/6] Creating environment configuration...${NC}"

if [ ! -f "$INSTALL_DIR/.env" ]; then
    cat > "$INSTALL_DIR/.env" << 'ENVFILE'
# AMYGDALA System - Environment Configuration
# ============================================

# Anthropic Claude API (for complex reasoning - prefrontal route)
ANTHROPIC_API_KEY=

# Hyperbolic API (for fast inference - limbic/cortical routes)
HYPERBOLIC_API_KEY=

# Server settings
AMYGDALA_HOST=0.0.0.0
AMYGDALA_PORT=8000

# Optional: Override default models
# AMYGDALA_CLAUDE_MODEL=claude-sonnet-4-20250514
# AMYGDALA_HYPERBOLIC_FAST=meta-llama/Meta-Llama-3.1-70B-Instruct
# AMYGDALA_HYPERBOLIC_REASONING=deepseek-ai/DeepSeek-V3

# Optional: Routing thresholds (0.0 to 1.0)
# AMYGDALA_THRESHOLD_LOW=0.3
# AMYGDALA_THRESHOLD_HIGH=0.7
ENVFILE

    echo -e "${YELLOW}⚠️  Created .env file. Please edit it with your API keys:${NC}"
    echo "   nano $INSTALL_DIR/.env"
else
    echo "✅ .env file already exists"
fi

# =============================================================================
# STEP 6: Create Systemd Service
# =============================================================================
echo -e "${GREEN}[6/6] Creating systemd service...${NC}"

cat > /etc/systemd/system/amygdala.service << SERVICEFILE
[Unit]
Description=AMYGDALA Neural System
After=network.target

[Service]
Type=simple
User=$AMYGDALA_USER
Group=$AMYGDALA_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/python server.py
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR

[Install]
WantedBy=multi-user.target
SERVICEFILE

# Set permissions
chown -R "$AMYGDALA_USER:$AMYGDALA_USER" "$INSTALL_DIR"

# Reload systemd
systemctl daemon-reload

echo "✅ Systemd service created"

# =============================================================================
# COMPLETION
# =============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    AMYGDALA INSTALLATION COMPLETE!                             ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "📋 Next steps:"
echo ""
echo "   1. Copy your source files to $INSTALL_DIR:"
echo "      - amygdala_core.py"
echo "      - server.py"
echo "      - AMYGDALA_FINAL.html"
echo ""
echo "   2. Configure API keys:"
echo "      nano $INSTALL_DIR/.env"
echo ""
echo "   3. Start the service:"
echo "      systemctl start amygdala"
echo "      systemctl enable amygdala"
echo ""
echo "   4. Check status:"
echo "      systemctl status amygdala"
echo "      journalctl -u amygdala -f"
echo ""
echo "   5. Access the interface:"
echo "      http://YOUR_SERVER_IP:$AMYGDALA_PORT"
echo ""
echo "   6. API documentation:"
echo "      http://YOUR_SERVER_IP:$AMYGDALA_PORT/docs"
echo ""
echo -e "${GREEN}🧠 AMYGDALA is ready to think!${NC}"
