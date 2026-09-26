CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -pthread
LLAMA_CPP ?= ../../llama.cpp
GGML_LIB_DIR ?= $(LLAMA_CPP)/build/bin

UNAME_S := $(shell uname -s)
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
endif

.PHONY: all clean cpu metal agent tools codenav \
	test-vectors test-model-profiles test-quant-k test-agent-bash test-agent-checkpoint test-agent-checkpoint-compatibility test-agent-coding-smoke test-agent-context test-agent-edit test-agent-prefix-cache test-agent-repeat test-agent-trace test-agent-verify test-metal-smoke test-metal-logits test-metal-logits-concurrent test-metal-gqa-decode \
	test-dense-forward test-dense-logits test-dense-session test-dense-metal-bringup test-dense-agent test-metal-session-roundtrip test-runtime-fingerprint test-runtime-regression test-prefill-bench test-regression test-regression-full \
	qw3-metal qw3-bench-metal qw3-eval-metal

all: qw3-cli qw3-agent qw3-eval qw3-bench
agent: qw3-agent
cpu: qw3-cpu qw3-agent-cpu qw3-eval-cpu qw3-bench-cpu

qw3_cpu_core.o: qw3.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3.c

qw3_cpu_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_cli.c

qw3_cpu_cli_test.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_cpu_agent.o: qw3_agent.c qw3.h linenoise.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_agent.c

qw3_eval_cpu.o: qw3_eval.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_eval.c

qw3_bench.o: qw3_bench.c qw3.h
	$(CC) $(CFLAGS) -c -o $@ qw3_bench.c

linenoise_qw3.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

qw3-cpu: qw3_cpu_cli.o qw3_cpu_core.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_cli.o qw3_cpu_core.o $(LDLIBS)

qw3-cpu-test: qw3_cpu_cli_test.o qw3_cpu_core.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_cli_test.o qw3_cpu_core.o $(LDLIBS)

qw3-bench-cpu: qw3_bench.o qw3_cpu_core.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3_cpu_core.o $(LDLIBS)

qw3-agent-cpu: qw3_cpu_agent.o qw3_cpu_core.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_agent.o qw3_cpu_core.o linenoise_qw3.o $(LDLIBS)

qw3-eval-cpu: qw3_eval_cpu.o qw3_cpu_core.o
	$(CC) $(CFLAGS) -o $@ qw3_eval_cpu.o qw3_cpu_core.o $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
metal: qw3-cli qw3-agent qw3-eval qw3-bench

.PHONY: FORCE
FORCE:

qw3_runtime_fingerprint.h: FORCE
	python3 tools/runtime_fingerprint.py $@ '$(CC)' '$(CFLAGS)' '$(OBJCFLAGS)' '$(METAL_LDLIBS)'

qw3_metal_core.o: qw3.c qw3.h qw3_metal.h qw3_runtime_fingerprint.h
	$(CC) $(CFLAGS) -c -o $@ qw3.c

qw3_metal_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -c -o $@ qw3_cli.c

qw3_metal_cli_test.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_metal_agent.o: qw3_agent.c qw3.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ qw3_agent.c

qw3_eval_metal.o: qw3_eval.c qw3.h qw3_metal.h
	$(CC) $(CFLAGS) -c -o $@ qw3_eval.c

qw3_metal.o: qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -c -o $@ qw3_metal.m

qw3-cli: qw3_metal_cli.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-test: qw3_metal_cli_test.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli_test.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-bench: qw3_bench.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-agent: qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o $(METAL_LDLIBS)

qw3-eval: qw3_eval_metal.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_eval_metal.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-metal: qw3-cli
	cp qw3-cli qw3-metal

qw3-bench-metal: qw3-bench
	cp qw3-bench qw3-bench-metal

qw3-eval-metal: qw3-eval
	cp qw3-eval qw3-eval-metal
else
metal:
	@echo "Metal backend requires Darwin/Apple Metal"
	@exit 1

qw3-cli: qw3-cpu
	cp qw3-cpu qw3-cli

qw3-bench: qw3-bench-cpu
	cp qw3-bench-cpu qw3-bench

qw3-agent: qw3-agent-cpu
	cp qw3-agent-cpu qw3-agent

qw3-eval: qw3-eval-cpu
	cp qw3-eval-cpu qw3-eval

qw3-metal qw3-bench-metal qw3-eval-metal:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1
endif

tools codenav:
	$(MAKE) -C codenavsrc

test-vectors: qw3-cpu-test
	QW3_BIN=./qw3-cpu-test sh tests/test_vectors.sh

test-model-profiles: qw3-cpu qw3-cpu-test
	sh tests/test_model_profiles.sh

qw3-quant-k-test: tests/test_quant_k.c qw3.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -o $@ tests/test_quant_k.c \
		-L$(GGML_LIB_DIR) -Wl,-rpath,$(abspath $(GGML_LIB_DIR)) \
		-lggml-cpu -lggml-base -lggml $(LDLIBS)

test-quant-k: qw3-quant-k-test
	./qw3-quant-k-test

test-dense-forward: qw3-cpu-test
	sh tests/test_dense_forward.sh

qw3-dense-logits-test: tests/test_dense_logits.c qw3_cpu_core.o qw3.h
	$(CC) $(filter-out -std=c99,$(CFLAGS)) -std=c11 \
		-I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include \
		-o $@ tests/test_dense_logits.c qw3_cpu_core.o \
		-L$(GGML_LIB_DIR) -Wl,-rpath,$(abspath $(GGML_LIB_DIR)) \
		-lllama -lggml-cpu -lggml-base -lggml $(LDLIBS)

test-dense-logits: qw3-dense-logits-test
	./qw3-dense-logits-test \
		$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf} 66

test-dense-session: qw3-cpu-test
	sh tests/test_dense_session.sh

test-dense-metal-bringup: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_dense_metal_bringup.sh

test-dense-agent: qw3-agent
	sh tests/test_dense_agent.sh

test-agent-edit: qw3-agent
	sh tests/test_agent_edit_tool.sh

test-agent-bash: qw3-agent
	sh tests/test_agent_bash_tool.sh

test-agent-checkpoint: qw3-agent
	sh tests/test_agent_checkpoint.sh

test-agent-coding-smoke: qw3-agent
	sh tests/test_agent_coding_smoke.sh

test-agent-prefix-cache: qw3-agent
	sh tests/test_agent_prefix_cache.sh

test-agent-color: qw3-agent
	QW3_AGENT_COLOR_SELFTEST=1 ./qw3-agent

test-agent-context: qw3-agent
	QW3_AGENT_CONTEXT_LEDGER_SELFTEST=1 ./qw3-agent

test-agent-repeat: qw3-agent
	QW3_AGENT_REPEAT_GUARD_SELFTEST=1 ./qw3-agent

test-agent-status: qw3-agent
	QW3_AGENT_STATUS_SELFTEST=1 ./qw3-agent

test-agent-trace: qw3-agent
	sh tests/test_agent_trace.sh

test-agent-tool-parse: qw3-agent
	QW3_AGENT_TOOL_PARSE_SELFTEST=1 ./qw3-agent

test-agent-verify: qw3-agent
	QW3_AGENT_VERIFY_GUARD_SELFTEST=1 ./qw3-agent

test-metal-smoke: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_smoke.sh

test-metal-logits: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_logits_regression.sh

test-metal-gqa-decode: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_gqa_decode.sh

test-metal-logits-concurrent: qw3-test
	QW3_METAL_BIN=./qw3-test QW3_METAL_PREFILL_CONCURRENT=1 sh tests/test_metal_logits_regression.sh

test-metal-session-roundtrip: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_session_roundtrip.sh

test-agent-checkpoint-compatibility: qw3-agent
	sh tests/test_agent_checkpoint_compatibility.sh

test-runtime-fingerprint:
	python3 tests/test_runtime_fingerprint.py

test-runtime-regression: qw3-cli qw3-agent
	sh tests/test_runtime_regression.sh

test-prefill-bench: qw3-bench
	sh tests/test_prefill_bench_regression.sh

test-regression: test-agent-bash test-agent-checkpoint test-agent-checkpoint-compatibility test-agent-color test-agent-context test-agent-edit test-agent-prefix-cache test-agent-repeat test-agent-status test-agent-tool-parse test-agent-trace test-agent-verify test-metal-logits test-metal-session-roundtrip test-runtime-fingerprint test-runtime-regression

.PHONY: test-metal-sampling
test-metal-sampling:
	sh tests/test_metal_sampling.sh

test-regression-full: test-regression test-agent-coding-smoke test-metal-smoke test-metal-gqa-decode test-metal-sampling

clean:
	rm -f qw3_runtime_fingerprint.h
	rm -f qw3 qw3-cli qw3-cpu qw3-test qw3-cpu-test qw3-quant-k-test qw3-dense-logits-test qw3-metal qw3-agent qw3-agent-cpu \
		qw3-bench qw3-bench-cpu qw3-bench-metal \
		qw3-eval qw3-eval-cpu qw3-eval-metal *.o
