# bonsai - MLX-powered LLM server
# Usage: make setup && make start

SHELL       := /bin/zsh
VENV        := .venv
UV          := uv
PYTHON      := $(VENV)/bin/python
PID_FILE    := .bonsai.pid
LOG_FILE    := bonsai.log
HOST        := 127.0.0.1
PORT        := 8430
MODEL       := prism-ml/Bonsai-8B-mlx-1bit

.PHONY: setup start stop status log test test-tools bench generate download clean

setup: _install_uv _ensure_metal_toolchain _venv _deps download
	@echo "\n[OK]  Setup complete. Run 'make start' to launch the server."

_install_uv:
	@if ! command -v $(UV) &>/dev/null; then \
		echo "=> Installing uv ..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "=> uv already installed: $$($(UV) --version)"; \
	fi

_ensure_metal_toolchain:
	@if ! xcrun metal -v &>/dev/null 2>&1; then \
		echo "=> Metal Toolchain missing - downloading (may need sudo) ..."; \
		xcodebuild -downloadComponent MetalToolchain; \
	else \
		echo "=> Metal Toolchain present."; \
	fi

_venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "=> Creating virtual environment ..."; \
		$(UV) venv $(VENV); \
	else \
		echo "=> Virtual environment exists."; \
	fi

_deps:
	@echo "=> Installing Python dependencies ..."
	$(UV) pip install --quiet mlx-lm
	@echo "=> Installing PrismML MLX fork (1-bit quant + Metal space-path fix) ..."
	$(UV) pip install --quiet ./mlx

download:
	@echo "=> Pre-downloading model $(MODEL) ..."
	$(VENV)/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)')"
	@echo "=> Model cached."

start:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "=> Starting bonsai mlx_lm.server on $(HOST):$(PORT) ..."; \
		$(VENV)/bin/python -m mlx_lm.server \
			--model $(MODEL) \
			--host $(HOST) --port $(PORT) \
			--temp 0.5 --top-p 0.85 \
			>> $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "=> Server PID: $$(cat $(PID_FILE))  (log: $(LOG_FILE))"; \
		echo "=> Waiting for server to be ready ..."; \
		for i in {1..60}; do \
			if curl -sf http://$(HOST):$(PORT)/v1/models >/dev/null 2>&1; then \
				echo "=> Server is ready."; \
				break; \
			fi; \
			sleep 2; \
		done; \
	fi

stop:
	@if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "=> Stopping server (PID $$PID) ..."; \
			kill $$PID; \
			rm -f $(PID_FILE); \
			echo "=> Stopped."; \
		else \
			echo "=> Stale PID file - removing."; \
			rm -f $(PID_FILE); \
		fi; \
	else \
		echo "=> No PID file found; server not running."; \
	fi

# -- status -----------------------------------------------------------
status:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Process: running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "Process: not running"; \
	fi
	@curl -sf http://$(HOST):$(PORT)/v1/models 2>/dev/null \
		&& echo "" \
		|| echo "Server: unreachable"

log:
	@if [ -f "$(LOG_FILE)" ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file yet."; \
	fi

test:
	@bash test.sh
	@echo "\n=> Running Python tool calling tests ..."
	$(VENV)/bin/python test_tools.py
	@echo "\n=> Running TypeScript tool calling tests ..."
	bunx tsx test_tools.ts

# -- bench ------------------------------------------------------------
bench:
	@bash bench.sh

# -- clean ------------------------------------------------------------
clean:
	@echo "=> Removing virtual environment and cached state ..."
	rm -rf $(VENV) $(PID_FILE) $(LOG_FILE)
	@echo "=> Clearing uv git cache (PrismML fork) ..."
	rm -rf $$(python3 -c "import pathlib; p=pathlib.Path.home()/'.cache'/'uv'/'git-v0'; print(p)" 2>/dev/null)
	@echo "=> Clean complete. Run 'make setup' to reinstall."
