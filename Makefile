# bonsai - MLX-powered LLM server
# Usage: make setup && make start

SHELL       := /bin/zsh
VENV        := .venv
UV          := uv
PYTHON      := $(VENV)/bin/python
PID_FILE    := .bonsai.pid
LOG_FILE    := bonsai-proxy.log
HOST        := 127.0.0.1
PORT        := 8430
MODEL       := prism-ml/Bonsai-8B-mlx-1bit
DRAFT_MODEL :=
MAX_BACKENDS := 2
MAX_MEM_UTIL := 80

.PHONY: setup start stop status log test test-tools bench download patch unpatch clean models

setup: _install_uv _ensure_metal_toolchain _venv _deps patch download
	@echo "\n[OK] Setup complete. Run 'make start' to launch the server."

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
	$(UV) pip install --quiet 'mlx-lm==0.31.1' openai python-dotenv
	@echo "=> Installing PrismML MLX fork (1-bit quant + Metal space-path fix) ..."
	$(UV) pip install --quiet ./mlx

MLX_LM_DIR = $$($(PYTHON) -c "import mlx_lm; print(mlx_lm.__path__[0])")

patch: unpatch
	@echo "=> Applying rotation patch to mlx_lm ..."
	cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/1_rotation.patch
	@echo "=> Applying turbo quant patch to mlx_lm ..."
	cd $(MLX_LM_DIR) && patch -p2 --forward < $(CURDIR)/patches/2_turbo_quant.patch
	@find $(MLX_LM_DIR) -name '*.rej' -delete 2>/dev/null || true
	@find $(MLX_LM_DIR) -name '*.orig' -delete 2>/dev/null || true

download:
	@echo "=> Pre-downloading model $(MODEL) ..."
	$(VENV)/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)')"
	@if [ -n "$(DRAFT_MODEL)" ]; then \
		echo "=> Pre-downloading draft model $(DRAFT_MODEL) ..."; \
		$(VENV)/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(DRAFT_MODEL)')"; \
	fi
	@echo "=> Models cached."

# To enable TurboQuant KV cache, add after --max-tokens:
#   --turbo-kv-bits 8 --turbo-fp16-layers 2
start:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Proxy already running (PID $$(cat $(PID_FILE)))"; \
	else \
		echo "=> Starting bonsai-proxy on $(HOST):$(PORT) ..."; \
		$(PYTHON) proxy.py \
			--host $(HOST) --port $(PORT) \
			--model $(MODEL) \
			$(if $(DRAFT_MODEL),--draft-model $(DRAFT_MODEL)) \
			--max-backends $(MAX_BACKENDS) \
			--max-mem-util $(MAX_MEM_UTIL) \
			--temp 0.5 --top-p 0.85 \
			--kv-bits 8 \
			--quantized-kv-start 128 \
			--max-tokens 65536 \
			>> $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "=> Proxy PID: $$(cat $(PID_FILE))  (log: $(LOG_FILE))"; \
		echo "=> Waiting for proxy + backend to be ready ..."; \
		for i in {1..60}; do \
			if curl -sf http://$(HOST):$(PORT)/v1/models >/dev/null 2>&1; then \
				echo "=> Ready."; \
				break; \
			fi; \
			sleep 2; \
		done; \
	fi

stop:
	@if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "=> Stopping proxy (PID $$PID) ..."; \
			kill $$PID; \
			for i in {1..20}; do \
				kill -0 $$PID 2>/dev/null || break; \
				sleep 0.25; \
			done; \
			if kill -0 $$PID 2>/dev/null; then \
				echo "=> Proxy didn't exit, sending SIGKILL ..."; \
				kill -9 $$PID 2>/dev/null; \
			fi; \
		else \
			echo "=> Proxy already dead (stale PID file)."; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "=> No PID file found."; \
	fi
	@# Kill any orphaned mlx_lm.server backends
	@ORPHANS=$$(pgrep -f 'mlx_lm\.server.*--host 127\.0\.0\.1' 2>/dev/null || true); \
	if [ -n "$$ORPHANS" ]; then \
		echo "=> Killing orphaned backends: $$ORPHANS"; \
		echo "$$ORPHANS" | xargs kill 2>/dev/null; \
		sleep 1; \
		echo "$$ORPHANS" | xargs kill -9 2>/dev/null || true; \
	fi
	@echo "=> Stopped."

# -- status -----------------------------------------------------------
status:
	@if [ -f "$(PID_FILE)" ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Proxy: running (PID $$(cat $(PID_FILE)))"; \
		curl -sf http://$(HOST):$(PORT)/health 2>/dev/null | python3 -m json.tool || true; \
	else \
		echo "Proxy: not running"; \
	fi

log:
	@if [ -f "$(LOG_FILE)" ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file yet."; \
	fi

test:
	@echo "=> Running integration tests ..."
	$(VENV)/bin/python test.py
	@echo "\n=> Running tool calling tests ..."
	$(VENV)/bin/python test_tools.py

# -- bench ------------------------------------------------------------
bench:
	@$(VENV)/bin/python bench.py

# -- models -----------------------------------------------------------
models:
	@$(PYTHON) -c "\
	from huggingface_hub import scan_cache_dir; \
	info = scan_cache_dir(); \
	models = sorted([ \
	    (r.repo_id, r.size_on_disk, str(r.repo_path)) \
	    for r in info.repos if r.repo_type == 'model' \
	]); \
	wn = max((len(r) for r, _, _ in models), default=20); \
	wp = max((len(p) for _, _, p in models), default=20); \
	print(f'{\"Model\":<{wn}}  {\"Size\":>8}  Location'); \
	print('-' * (wn + wp + 14)); \
	[print(f'{rid:<{wn}}  {sz/1e9:>7.2f}G  {path}') for rid, sz, path in models]; \
	print(f'\n{len(models)} model(s), {sum(sz for _,sz,_ in models)/1e9:.2f} GB total') \
	"

# -- unpatch ----------------------------------------------------------
unpatch:
	@echo "=> Reinstalling mlx-lm (clean, unpatched) ..."
	$(UV) pip install --quiet --force-reinstall --no-deps 'mlx-lm==0.31.1'
	@echo "=> Removing leftover patch artifacts ..."
	@find $(MLX_LM_DIR)/models -name 'turboquant_*.py' -delete 2>/dev/null || true
	@rm -f $(MLX_LM_DIR)/test_turboquant.py 2>/dev/null || true
	@find $(MLX_LM_DIR) -name '*.rej' -delete 2>/dev/null || true
	@find $(MLX_LM_DIR) -name '*.orig' -delete 2>/dev/null || true
	@echo "=> mlx_lm restored to clean state."

# -- clean ------------------------------------------------------------
clean:
	@echo "=> Removing virtual environment and cached state ..."
	rm -rf $(VENV) $(PID_FILE) $(LOG_FILE) bonsai-backend-*.log
	@echo "=> Clearing uv git cache (PrismML fork) ..."
	rm -rf $$(python3 -c "import pathlib; p=pathlib.Path.home()/'.cache'/'uv'/'git-v0'; print(p)" 2>/dev/null)
	@echo "=> Clean complete. Run 'make setup' to reinstall."
