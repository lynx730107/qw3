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

.PHONY: all clean cpu metal agent tools codenav bench-sampled-decode test-agent-git-smoke \
	test-vectors test-model-profiles test-quant-k test-agent-bash test-agent-checkpoint test-agent-checkpoint-compatibility test-agent-coding-smoke test-agent-python-smoke test-agent-context test-agent-edit test-agent-llama-reference test-agent-prefix-cache test-agent-repeat test-agent-trace test-agent-verify test-agent-web-moe test-agent-web-think test-metal-smoke test-metal-logits test-metal-logits-concurrent test-metal-gqa-decode \
	test-dense-forward test-dense-logits test-dense-session test-dense-metal-bringup test-dense-gqa-decode test-dense-agent test-metal-session-roundtrip test-runtime-fingerprint test-runtime-regression test-prefill-bench test-regression test-regression-full \
	qw3-metal qw3-bench-metal qw3-eval-metal

.PHONY: test-rope test-rope-metal test-dense-metal-logits
.PHONY: test-agent-repeat-format test-agent-store
.PHONY: test-agent-context-pressure
.PHONY: test-agent-web-page
.PHONY: test-short-batch-quant test-short-batch
.PHONY: test-iq3-stream-quant

all: qw3-cli qw3-agent qw3-eval qw3-bench
agent: qw3-agent
cpu: qw3-cpu qw3-agent-cpu qw3-eval-cpu qw3-bench-cpu

qw3_cpu_core.o: qw3.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3.c

qw3_cpu_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_cli.c

qw3_cpu_cli_test.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_cpu_agent.o: qw3_agent.c qw3.h qw3_agent_unicode.h linenoise.h
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

qw3_metal_agent.o: qw3_agent.c qw3.h qw3_agent_unicode.h linenoise.h
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

qw3-sampled-decode-bench: tests/bench_sampled_decode.c qw3.h qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ tests/bench_sampled_decode.c qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

# Diagnostic-only access to dispatch choices; no tuning API in production.
qw3-decode-ab-bench: tests/bench_decode_ab.m qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_metal_core.o
	$(CC) $(OBJCFLAGS) -o $@ tests/bench_decode_ab.m qw3_metal_core.o $(METAL_LDLIBS)

# Isolated experiments only; neither target is a CLI/agent dependency.
qw3-q5-head-bench: tests/bench_q5_head.m tests/q5_decode_candidates.metal qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_metal_core.o
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_q5_head.m qw3_metal_core.o $(METAL_LDLIBS)

qw3-iq3-stream-bench: tests/bench_iq3_stream.m tests/iq3_decode_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_iq3_stream.m $(METAL_LDLIBS)

qw3-decode-trials-bench: tests/bench_decode_trials.m tests/iq4_layout_probe.inc tests/iq4_prefetch_probe.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_decode_trials.m $(METAL_LDLIBS)

qw3-projection-cost-bench: tests/bench_projection_cost.m tests/projection_candidates.inc tests/projection_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_projection_cost.m $(METAL_LDLIBS)

qw3-projection-prefill-bench: tests/bench_projection_prefill.m tests/projection_candidates.inc tests/projection_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_projection_prefill.m $(METAL_LDLIBS)

qw3-projection-prefill-test: tests/test_projection_prefill.m tests/projection_candidates.inc tests/projection_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_projection_prefill.m $(METAL_LDLIBS)

qw3-scratch-prefill-bench: tests/bench_scratch_prefill.m tests/scratch_candidates.inc tests/scratch_matmul.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_scratch_prefill.m $(METAL_LDLIBS)

qw3-scratch-prefill-test: tests/test_scratch_prefill.m tests/scratch_candidates.inc tests/scratch_matmul.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_scratch_prefill.m $(METAL_LDLIBS)

qw3-scratch-matmul-test: tests/test_scratch_matmul.m tests/scratch_candidates.inc tests/scratch_matmul.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_scratch_matmul.m $(METAL_LDLIBS)

tests/.kstage_candidates/kernels.metal: tests/prepare_kstage_candidates.py tests/test_native_prefill.m tests/test_register_iq4.m metal/qw3_core_moe.metal qw3_metal.m
	python3 tests/prepare_kstage_candidates.py

tests/.iq3_groups/wrapper.inc: tests/prepare_iq3_groups.py tests/test_native_prefill.m qw3_metal.m
	python3 tests/prepare_iq3_groups.py

tests/.gdn_cache/kernels.metal: tests/prepare_gdn_cache.py tests/test_native_prefill.m tests/bench_iq3_groups.m metal/qw3_core_deltanet.metal qw3_metal.m
	python3 tests/prepare_gdn_cache.py

qw3-gdn-cache-test: tests/test_gdn_cache.m tests/gdn_cache_candidates.inc tests/.gdn_cache/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_cache.m $(METAL_LDLIBS)

qw3-native-gdn-cache-test: tests/test_native_gdn_cache.m tests/native_prefill_validation.inc tests/.gdn_cache/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_native_gdn_cache.m $(METAL_LDLIBS)

tests/.decode_profile/hooks.inc: tests/prepare_decode_profile.py tests/projection_candidates.inc tests/test_native_prefill.m
	python3 tests/prepare_decode_profile.py

tests/.iq4_loads/quality.inc: tests/prepare_iq4_loads.py tests/test_native_prefill.m tests/bench_iq3_groups.m
	python3 tests/prepare_iq4_loads.py

tests/.iq4_packed_lut/hooks.inc: tests/prepare_iq4_packed_lut.py tests/iq4_load_candidates.inc tests/test_iq4_loads.m tests/test_native_prefill.m tests/bench_iq3_groups.m $(METAL_SRCS)
	python3 tests/prepare_iq4_packed_lut.py

tests/.iq4_packed_lut/kernels.metal tests/.iq4_packed_lut/quality.inc tests/.iq4_packed_lut/matrix.inc tests/.iq4_packed_lut/bench.inc: tests/.iq4_packed_lut/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_packed_lut.py

qw3-iq4-packed-lut-test: tests/test_iq4_packed_lut.m tests/.iq4_packed_lut/hooks.inc tests/.iq4_packed_lut/matrix.inc tests/.iq4_packed_lut/quality.inc tests/.iq4_packed_lut/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_packed_lut.m $(METAL_LDLIBS)

qw3-iq4-packed-lut-bench: tests/bench_iq4_packed_lut.m tests/.iq4_packed_lut/hooks.inc tests/.iq4_packed_lut/quality.inc tests/.iq4_packed_lut/bench.inc tests/.iq4_packed_lut/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_packed_lut.m $(METAL_LDLIBS)

tests/.iq3_private/hooks.inc: tests/prepare_iq3_private.py tests/.iq3_layout/hooks.inc tests/.iq3_layout/quality.inc tests/.iq3_layout/bench.inc
	python3 tests/prepare_iq3_private.py

tests/.iq3_private/quality.inc tests/.iq3_private/bench.inc: tests/.iq3_private/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_private.py

qw3-iq3-private-test: tests/test_iq3_private.m tests/.iq3_private/hooks.inc tests/.iq3_private/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_private.m $(METAL_LDLIBS)

qw3-iq3-private-bench: tests/bench_iq3_private.m tests/.iq3_private/hooks.inc tests/.iq3_private/quality.inc tests/.iq3_private/bench.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_private.m $(METAL_LDLIBS)

qw3-iq3-private-native-bench: tests/bench_iq3_private_native.m tests/.iq3_private/hooks.inc tests/.iq3_private/quality.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_private_native.m $(METAL_LDLIBS)

tests/.iq3_layout/hooks.inc: tests/prepare_iq3_layout.py tests/iq4_load_candidates.inc tests/test_native_prefill.m tests/bench_iq3_groups.m metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_layout.py

tests/.iq3_layout/kernels.metal tests/.iq3_layout/quality.inc tests/.iq3_layout/bench.inc: tests/.iq3_layout/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_layout.py

qw3-iq3-layout-test: tests/test_iq3_layout.m tests/.iq3_layout/hooks.inc tests/.iq3_layout/quality.inc tests/.iq3_layout/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_layout.m $(METAL_LDLIBS)

qw3-iq3-layout-bench: tests/bench_iq3_layout.m tests/.iq3_layout/hooks.inc tests/.iq3_layout/quality.inc tests/.iq3_layout/bench.inc tests/.iq3_layout/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_layout.m $(METAL_LDLIBS)

tests/.iq3_grid/hooks.inc: tests/prepare_iq3_grid.py tests/iq4_load_candidates.inc tests/test_iq3_signed.m metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_grid.py

tests/.q4_words/hooks.inc: tests/prepare_q4_words.py tests/.q4_rows/hooks.inc tests/.q4_rows/model.inc tests/.q4_rows/native.inc metal/qw3_core_linear.metal
	python3 tests/prepare_q4_words.py

tests/.q4_words/kernels.metal tests/.q4_words/model.inc tests/.q4_words/native.inc: tests/.q4_words/hooks.inc
	@test -f $@ || python3 tests/prepare_q4_words.py

qw3-q4-words-test: tests/test_q4_words.m tests/.q4_words/hooks.inc tests/.q4_words/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_words.m $(METAL_LDLIBS)

qw3-q4-words-model-test: tests/test_q4_words_model.m tests/.q4_words/hooks.inc tests/.q4_words/kernels.metal tests/.q4_words/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_words_model.m $(METAL_LDLIBS)

qw3-q4-words-native-bench: tests/bench_q4_words_native.m tests/.q4_words/hooks.inc tests/.q4_words/kernels.metal tests/.q4_words/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_q4_words_native.m $(METAL_LDLIBS)

tests/.q4_rows/hooks.inc: tests/prepare_q4_rows.py tests/.gqa_barrier/hooks.inc tests/.q4_narrow/model.inc tests/.q4_narrow/native.inc metal/qw3_core_linear.metal
	python3 tests/prepare_q4_rows.py

tests/.q4_rows/kernels.metal tests/.q4_rows/model.inc tests/.q4_rows/native.inc: tests/.q4_rows/hooks.inc
	@test -f $@ || python3 tests/prepare_q4_rows.py

qw3-q4-rows-test: tests/test_q4_rows.m tests/.q4_rows/hooks.inc tests/.q4_rows/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_rows.m $(METAL_LDLIBS)

qw3-q4-rows-model-test: tests/test_q4_rows_model.m tests/.q4_rows/hooks.inc tests/.q4_rows/kernels.metal tests/.q4_rows/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_rows_model.m $(METAL_LDLIBS)

qw3-q4-rows-native-bench: tests/bench_q4_rows_native.m tests/.q4_rows/hooks.inc tests/.q4_rows/kernels.metal tests/.q4_rows/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_q4_rows_native.m $(METAL_LDLIBS)

tests/.q4_narrow/hooks.inc: tests/prepare_q4_narrow.py qw3_metal.m tests/.iq3_words/model.inc tests/.iq3_words/native.inc
	python3 tests/prepare_q4_narrow.py

tests/.q4_narrow/core.inc tests/.q4_narrow/model.inc tests/.q4_narrow/native.inc: tests/.q4_narrow/hooks.inc
	@test -f $@ || python3 tests/prepare_q4_narrow.py

qw3-q4-narrow-test: tests/test_q4_narrow.m tests/.q4_narrow/hooks.inc tests/.q4_narrow/core.inc qw3.c qw3.h qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_narrow.m $(METAL_LDLIBS)

qw3-q4-narrow-model-test: tests/test_q4_narrow_model.m tests/.q4_narrow/hooks.inc tests/.q4_narrow/core.inc tests/.q4_narrow/model.inc qw3.c qw3.h qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q4_narrow_model.m $(METAL_LDLIBS)

qw3-q4-narrow-native-bench: tests/bench_q4_narrow_native.m tests/.q4_narrow/hooks.inc tests/.q4_narrow/core.inc tests/.q4_narrow/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_q4_narrow_native.m $(METAL_LDLIBS)

tests/.iq3_input/hooks.inc: tests/prepare_iq3_input.py tests/.iq3_words/hooks.inc tests/.iq3_words/matrix.inc tests/.iq3_words/model.inc tests/.iq3_words/native.inc metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_input.py

tests/.iq3_input/kernels.metal tests/.iq3_input/matrix.inc tests/.iq3_input/model.inc tests/.iq3_input/native.inc: tests/.iq3_input/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_input.py

tests/.iq3_words/hooks.inc: tests/prepare_iq3_words.py tests/.iq3_grid/hooks.inc tests/.iq3_grid/matrix.inc tests/test_iq3_grid_model.m tests/bench_iq3_grid_native.m metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_words.py

tests/.iq3_words/kernels.metal tests/.iq3_words/matrix.inc tests/.iq3_words/model.inc tests/.iq3_words/native.inc: tests/.iq3_words/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_words.py

tests/.iq3_hint/hooks.inc: tests/prepare_iq3_hint.py tests/.iq3_words/hooks.inc tests/.iq3_words/matrix.inc tests/.iq3_words/model.inc tests/.iq3_words/native.inc
	python3 tests/prepare_iq3_hint.py

tests/.iq3_hint/matrix.inc tests/.iq3_hint/model.inc tests/.iq3_hint/native.inc: tests/.iq3_hint/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_hint.py

tests/.iq3_immutable/hooks.inc: tests/prepare_iq3_immutable.py tests/.iq3_hint/hooks.inc tests/.iq3_hint/matrix.inc tests/.iq3_hint/model.inc tests/.iq3_hint/native.inc
	python3 tests/prepare_iq3_immutable.py

tests/.iq3_rows/hooks.inc: tests/prepare_iq3_rows.py tests/.iq3_immutable/hooks.inc tests/.iq3_immutable/matrix.inc tests/.iq3_immutable/model.inc tests/.iq3_immutable/native.inc qw3_metal.m metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_rows.py

tests/.iq3_share/hooks.inc: tests/prepare_iq3_share.py tests/.iq3_rows/hooks.inc tests/.iq3_rows/kernels.metal tests/.iq3_rows/matrix.inc tests/.iq3_rows/model.inc tests/.iq3_rows/native.inc qw3_metal.m
	python3 tests/prepare_iq3_share.py

tests/.iq4_share/hooks.inc: tests/prepare_iq4_share.py tests/.iq3_share/hooks.inc tests/.iq3_share/model.inc tests/.iq3_share/native.inc tests/test_iq4_loads.m qw3_metal.m metal/qw3_core_linear.metal
	python3 tests/prepare_iq4_share.py

tests/.iq4_share/core.inc tests/.iq4_share/kernels.metal tests/.iq4_share/matrix.inc tests/.iq4_share/model.inc tests/.iq4_share/native.inc: tests/.iq4_share/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_share.py

tests/.gqa_barrier/hooks.inc: tests/prepare_gqa_barrier.py tests/.iq4_unroll/hooks.inc tests/.iq4_unroll/model.inc tests/.iq4_unroll/native.inc metal/qw3_core_sequence.metal
	python3 tests/prepare_gqa_barrier.py

tests/.gqa_barrier/kernels.metal tests/.gqa_barrier/model.inc tests/.gqa_barrier/native.inc: tests/.gqa_barrier/hooks.inc
	@test -f $@ || python3 tests/prepare_gqa_barrier.py

qw3-gqa-barrier-test: tests/test_gqa_barrier.m tests/.gqa_barrier/hooks.inc tests/.gqa_barrier/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gqa_barrier.m $(METAL_LDLIBS)

qw3-gqa-barrier-model-test: tests/test_gqa_barrier_model.m tests/.gqa_barrier/hooks.inc tests/.gqa_barrier/kernels.metal tests/.gqa_barrier/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gqa_barrier_model.m $(METAL_LDLIBS)

qw3-gqa-barrier-native-bench: tests/bench_gqa_barrier_native.m tests/.gqa_barrier/hooks.inc tests/.gqa_barrier/kernels.metal tests/.gqa_barrier/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gqa_barrier_native.m $(METAL_LDLIBS)

tests/.iq4_shape/hooks.inc: tests/prepare_iq4_shape.py tests/.iq4_unroll/hooks.inc tests/.iq4_unroll/matrix.inc tests/.iq4_unroll/model.inc tests/.iq4_unroll/native.inc metal/qw3_core_linear.metal
	python3 tests/prepare_iq4_shape.py

tests/.iq4_shape/kernels.metal tests/.iq4_shape/matrix.inc tests/.iq4_shape/model.inc tests/.iq4_shape/native.inc: tests/.iq4_shape/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_shape.py

qw3-iq4-shape-test: tests/test_iq4_shape.m tests/.iq4_shape/hooks.inc tests/.iq4_shape/kernels.metal tests/.iq4_shape/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_shape.m $(METAL_LDLIBS)

qw3-iq4-shape-model-test: tests/test_iq4_shape_model.m tests/.iq4_shape/hooks.inc tests/.iq4_shape/kernels.metal tests/.iq4_shape/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_shape_model.m $(METAL_LDLIBS)

qw3-iq4-shape-native-bench: tests/bench_iq4_shape_native.m tests/.iq4_shape/hooks.inc tests/.iq4_shape/kernels.metal tests/.iq4_shape/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_shape_native.m $(METAL_LDLIBS)

tests/.iq4_simd_lut/hooks.inc: tests/prepare_iq4_simd_lut.py tests/.iq4_share/hooks.inc tests/.iq4_share/matrix.inc tests/.iq4_share/model.inc tests/.iq4_share/native.inc metal/qw3_core_linear.metal
	python3 tests/prepare_iq4_simd_lut.py

tests/.iq4_simd_lut/kernels.metal tests/.iq4_simd_lut/matrix.inc tests/.iq4_simd_lut/model.inc tests/.iq4_simd_lut/native.inc: tests/.iq4_simd_lut/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_simd_lut.py

qw3-iq4-simd-lut-test: tests/test_iq4_simd_lut.m tests/.iq4_simd_lut/hooks.inc tests/.iq4_simd_lut/kernels.metal tests/.iq4_simd_lut/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_simd_lut.m $(METAL_LDLIBS)

qw3-iq4-simd-lut-model-test: tests/test_iq4_simd_lut_model.m tests/.iq4_simd_lut/hooks.inc tests/.iq4_simd_lut/kernels.metal tests/.iq4_simd_lut/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_simd_lut_model.m $(METAL_LDLIBS)

qw3-iq4-simd-lut-native-bench: tests/bench_iq4_simd_lut_native.m tests/.iq4_simd_lut/hooks.inc tests/.iq4_simd_lut/kernels.metal tests/.iq4_simd_lut/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_simd_lut_native.m $(METAL_LDLIBS)

tests/.iq4_unroll/hooks.inc: tests/prepare_iq4_unroll.py tests/.iq4_share/hooks.inc tests/.iq4_share/matrix.inc tests/.iq4_share/model.inc tests/.iq4_share/native.inc metal/qw3_core_linear.metal
	python3 tests/prepare_iq4_unroll.py

tests/.iq4_unroll/kernels.metal tests/.iq4_unroll/matrix.inc tests/.iq4_unroll/model.inc tests/.iq4_unroll/native.inc: tests/.iq4_unroll/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_unroll.py

qw3-iq4-unroll-test: tests/test_iq4_unroll.m tests/.iq4_unroll/hooks.inc tests/.iq4_unroll/kernels.metal tests/.iq4_unroll/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_unroll.m $(METAL_LDLIBS)

qw3-iq4-unroll-model-test: tests/test_iq4_unroll_model.m tests/.iq4_unroll/hooks.inc tests/.iq4_unroll/kernels.metal tests/.iq4_unroll/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_unroll_model.m $(METAL_LDLIBS)

qw3-iq4-unroll-native-bench: tests/bench_iq4_unroll_native.m tests/.iq4_unroll/hooks.inc tests/.iq4_unroll/kernels.metal tests/.iq4_unroll/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_unroll_native.m $(METAL_LDLIBS)

qw3-iq4-share-test: tests/test_iq4_share.m tests/.iq4_share/hooks.inc tests/.iq4_share/core.inc tests/.iq4_share/kernels.metal tests/.iq4_share/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_share.m $(METAL_LDLIBS)

qw3-iq4-share-model-test: tests/test_iq4_share_model.m tests/.iq4_share/hooks.inc tests/.iq4_share/core.inc tests/.iq4_share/kernels.metal tests/.iq4_share/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_share_model.m $(METAL_LDLIBS)

qw3-iq4-share-native-bench: tests/bench_iq4_share_native.m tests/.iq4_share/hooks.inc tests/.iq4_share/core.inc tests/.iq4_share/kernels.metal tests/.iq4_share/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_share_native.m $(METAL_LDLIBS)

tests/.iq3_share/core.inc tests/.iq3_share/kernels.metal tests/.iq3_share/matrix.inc tests/.iq3_share/model.inc tests/.iq3_share/native.inc: tests/.iq3_share/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_share.py

qw3-iq3-share-test: tests/test_iq3_share.m tests/.iq3_share/hooks.inc tests/.iq3_share/core.inc tests/.iq3_share/kernels.metal tests/.iq3_share/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_share.m $(METAL_LDLIBS)

qw3-iq3-share-model-test: tests/test_iq3_share_model.m tests/.iq3_share/hooks.inc tests/.iq3_share/core.inc tests/.iq3_share/kernels.metal tests/.iq3_share/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_share_model.m $(METAL_LDLIBS)

qw3-iq3-share-native-bench: tests/bench_iq3_share_native.m tests/.iq3_share/hooks.inc tests/.iq3_share/core.inc tests/.iq3_share/kernels.metal tests/.iq3_share/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_share_native.m $(METAL_LDLIBS)

tests/.iq3_rows/core.inc tests/.iq3_rows/kernels.metal tests/.iq3_rows/matrix.inc tests/.iq3_rows/model.inc tests/.iq3_rows/native.inc: tests/.iq3_rows/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_rows.py

qw3-iq3-rows-test: tests/test_iq3_rows.m tests/.iq3_rows/hooks.inc tests/.iq3_rows/core.inc tests/.iq3_rows/kernels.metal tests/.iq3_rows/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_rows.m $(METAL_LDLIBS)

qw3-iq3-rows-model-test: tests/test_iq3_rows_model.m tests/.iq3_rows/hooks.inc tests/.iq3_rows/core.inc tests/.iq3_rows/kernels.metal tests/.iq3_rows/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_rows_model.m $(METAL_LDLIBS)

qw3-iq3-rows-native-bench: tests/bench_iq3_rows_native.m tests/.iq3_rows/hooks.inc tests/.iq3_rows/core.inc tests/.iq3_rows/kernels.metal tests/.iq3_rows/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_rows_native.m $(METAL_LDLIBS)

tests/.gdn_two128/kernels.metal: tests/prepare_gdn_two128.py metal/qw3_core_deltanet.metal tests/.iq3_immutable/model.inc tests/.iq3_immutable/native.inc
	python3 tests/prepare_gdn_two128.py

tests/.gdn_two128/model.inc tests/.gdn_two128/native.inc: tests/.gdn_two128/kernels.metal
	@test -f $@ || python3 tests/prepare_gdn_two128.py

qw3-gdn-two128-test: tests/test_gdn_two128.m tests/gdn_two128_candidates.inc tests/.gdn_two128/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_two128.m $(METAL_LDLIBS)

qw3-gdn-two128-model-test: tests/test_gdn_two128_model.m tests/gdn_two128_candidates.inc tests/.gdn_two128/kernels.metal tests/.gdn_two128/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_two128_model.m $(METAL_LDLIBS)

qw3-gdn-two128-native-bench: tests/bench_gdn_two128_native.m tests/gdn_two128_candidates.inc tests/.gdn_two128/kernels.metal tests/.gdn_two128/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gdn_two128_native.m $(METAL_LDLIBS)

tests/.iq3_immutable/matrix.inc tests/.iq3_immutable/model.inc tests/.iq3_immutable/native.inc: tests/.iq3_immutable/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_immutable.py

qw3-iq3-immutable-test: tests/test_iq3_immutable.m tests/.iq3_immutable/hooks.inc tests/.iq3_immutable/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_immutable.m $(METAL_LDLIBS)

qw3-iq3-immutable-model-test: tests/test_iq3_immutable_model.m tests/.iq3_immutable/hooks.inc tests/.iq3_immutable/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_immutable_model.m $(METAL_LDLIBS)

qw3-iq3-immutable-native-bench: tests/bench_iq3_immutable_native.m tests/.iq3_immutable/hooks.inc tests/.iq3_immutable/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_immutable_native.m $(METAL_LDLIBS)

qw3-iq3-hint-test: tests/test_iq3_hint.m tests/.iq3_hint/hooks.inc tests/.iq3_hint/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_hint.m $(METAL_LDLIBS)

qw3-iq3-hint-model-test: tests/test_iq3_hint_model.m tests/.iq3_hint/hooks.inc tests/.iq3_hint/model.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_hint_model.m $(METAL_LDLIBS)

qw3-iq3-hint-native-bench: tests/bench_iq3_hint_native.m tests/.iq3_hint/hooks.inc tests/.iq3_hint/native.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_hint_native.m $(METAL_LDLIBS)

qw3-iq3-input-test: tests/test_iq3_input.m tests/.iq3_input/hooks.inc tests/.iq3_input/matrix.inc tests/.iq3_input/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_input.m $(METAL_LDLIBS)

qw3-iq3-input-model-test: tests/test_iq3_input_model.m tests/.iq3_input/hooks.inc tests/.iq3_input/model.inc tests/.iq3_input/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_input_model.m $(METAL_LDLIBS)

qw3-iq3-input-native-bench: tests/bench_iq3_input_native.m tests/.iq3_input/hooks.inc tests/.iq3_input/native.inc tests/.iq3_input/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_input_native.m $(METAL_LDLIBS)

qw3-iq3-shape-test: tests/test_iq3_shape.m tests/.iq3_shape/hooks.inc tests/.iq3_shape/matrix.inc tests/.iq3_shape/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_shape.m $(METAL_LDLIBS)

qw3-iq3-shape-model-test: tests/test_iq3_shape_model.m tests/.iq3_shape/hooks.inc tests/.iq3_shape/model.inc tests/.iq3_shape/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_shape_model.m $(METAL_LDLIBS)

qw3-iq3-shape-native-bench: tests/bench_iq3_shape_native.m tests/.iq3_shape/hooks.inc tests/.iq3_shape/native.inc tests/.iq3_shape/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_shape_native.m $(METAL_LDLIBS)

qw3-q5-words-test: tests/test_q5_words.m tests/.q5_words/hooks.inc tests/.q5_words/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q5_words.m $(METAL_LDLIBS)

qw3-q5-words-model-test: tests/test_q5_words_model.m tests/.q5_words/hooks.inc tests/.q5_words/model.inc tests/.q5_words/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_q5_words_model.m $(METAL_LDLIBS)

qw3-q5-words-native-bench: tests/bench_q5_words_native.m tests/.q5_words/hooks.inc tests/.q5_words/native.inc tests/.q5_words/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_q5_words_native.m $(METAL_LDLIBS)

qw3-iq3-add-test: tests/test_iq3_add.m tests/iq3_add_hooks.inc tests/.iq3_add/core.inc tests/.iq3_add/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_add.m $(METAL_LDLIBS)

qw3-iq3-add-model-test: tests/test_iq3_add_model.m tests/iq3_add_hooks.inc tests/.iq3_add/core.inc tests/.iq3_add/model.inc tests/.iq3_add/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_add_model.m $(METAL_LDLIBS)

qw3-iq3-add-native-bench: tests/bench_iq3_add_native.m tests/iq3_add_hooks.inc tests/.iq3_add/core.inc tests/.iq3_add/native.inc tests/.iq3_add/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_add_native.m $(METAL_LDLIBS)

qw3-iq3-pair-grid-test: tests/test_iq3_pair_grid.m tests/.iq3_pair_grid/hooks.inc tests/.iq3_pair_grid/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_pair_grid.m $(METAL_LDLIBS)

qw3-iq3-pair-grid-model-test: tests/test_iq3_pair_grid_model.m tests/.iq3_pair_grid/hooks.inc tests/.iq3_pair_grid/model.inc tests/.iq3_pair_grid/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_pair_grid_model.m $(METAL_LDLIBS)

qw3-iq3-pair-grid-native-bench: tests/bench_iq3_pair_grid_native.m tests/.iq3_pair_grid/hooks.inc tests/.iq3_pair_grid/native.inc tests/.iq3_pair_grid/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_pair_grid_native.m $(METAL_LDLIBS)

qw3-iq3-words-test: tests/test_iq3_words.m tests/.iq3_words/hooks.inc tests/.iq3_words/matrix.inc tests/.iq3_words/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_words.m $(METAL_LDLIBS)

qw3-iq3-words-model-test: tests/test_iq3_words_model.m tests/.iq3_words/hooks.inc tests/.iq3_words/model.inc tests/.iq3_words/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_words_model.m $(METAL_LDLIBS)

qw3-iq3-words-native-bench: tests/bench_iq3_words_native.m tests/.iq3_words/hooks.inc tests/.iq3_words/native.inc tests/.iq3_words/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_words_native.m $(METAL_LDLIBS)

tests/.iq3_grid/kernels.metal tests/.iq3_grid/matrix.inc: tests/.iq3_grid/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_grid.py

qw3-iq3-grid-test: tests/test_iq3_grid.m tests/.iq3_grid/hooks.inc tests/.iq3_grid/matrix.inc tests/.iq3_grid/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_grid.m $(METAL_LDLIBS)

qw3-iq3-grid-model-test: tests/test_iq3_grid_model.m tests/.iq3_grid/hooks.inc tests/.iq3_grid/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_grid_model.m $(METAL_LDLIBS)

qw3-iq3-grid-native-bench: tests/bench_iq3_grid_native.m tests/.iq3_grid/hooks.inc tests/.iq3_grid/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_grid_native.m $(METAL_LDLIBS)

tests/.iq3_signed/hooks.inc: tests/prepare_iq3_signed.py tests/iq4_load_candidates.inc tests/test_native_prefill.m tests/bench_iq3_groups.m metal/qw3_core_moe.metal
	python3 tests/prepare_iq3_signed.py

tests/.iq3_signed/kernels.metal tests/.iq3_signed/quality.inc tests/.iq3_signed/bench.inc: tests/.iq3_signed/hooks.inc
	@test -f $@ || python3 tests/prepare_iq3_signed.py

qw3-iq3-signed-test: tests/test_iq3_signed.m tests/.iq3_signed/hooks.inc tests/.iq3_signed/quality.inc tests/.iq3_signed/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_signed.m $(METAL_LDLIBS)

qw3-iq3-signed-bench: tests/bench_iq3_signed.m tests/.iq3_signed/hooks.inc tests/.iq3_signed/quality.inc tests/.iq3_signed/bench.inc tests/.iq3_signed/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_signed.m $(METAL_LDLIBS)

tests/.iq4_rows/hooks.inc: tests/prepare_iq4_rows.py tests/iq4_load_candidates.inc tests/test_iq4_loads.m tests/test_native_prefill.m tests/bench_iq3_groups.m $(METAL_SRCS)
	python3 tests/prepare_iq4_rows.py

tests/.iq4_rows/kernels.metal tests/.iq4_rows/quality.inc tests/.iq4_rows/matrix.inc tests/.iq4_rows/bench.inc: tests/.iq4_rows/hooks.inc
	@test -f $@ || python3 tests/prepare_iq4_rows.py

qw3-iq4-rows-test: tests/test_iq4_rows.m tests/.iq4_rows/hooks.inc tests/.iq4_rows/matrix.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_rows.m $(METAL_LDLIBS)

qw3-iq4-rows-bench: tests/bench_iq4_rows.m tests/.iq4_rows/hooks.inc tests/.iq4_rows/bench.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_rows.m $(METAL_LDLIBS)

qw3-iq4-rows-cost-bench: tests/bench_iq4_rows_cost.m tests/.iq4_rows/hooks.inc tests/.iq4_rows/quality.inc tests/.iq4_rows/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_rows_cost.m $(METAL_LDLIBS)

qw3-iq4-loads-test: tests/test_iq4_loads.m tests/iq4_load_candidates.inc tests/.iq4_loads/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq4_loads.m $(METAL_LDLIBS)

tests/.iq4_loads/bench.inc: tests/.iq4_loads/quality.inc
	@test -f $@ || python3 tests/prepare_iq4_loads.py

qw3-iq4-loads-bench: tests/bench_iq4_loads.m tests/iq4_load_candidates.inc tests/.iq4_loads/bench.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq4_loads.m $(METAL_LDLIBS)

qw3-decode-profile-test: tests/test_decode_profile.m tests/.decode_profile/hooks.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_decode_profile.m $(METAL_LDLIBS)

tests/.hazard_fences/quality.inc: tests/prepare_hazard_fences.py tests/prepare_hazard_tracking.py tests/test_native_prefill.m tests/native_prefill_validation.inc qw3_metal.m
	python3 tests/prepare_hazard_fences.py

qw3-hazard-fences-native-bench: tests/bench_hazard_fences_native.m tests/.hazard_fences/quality.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_hazard_fences_native.m $(METAL_LDLIBS)

qw3-hazard-fences-test: tests/test_hazard_fences.m tests/.hazard_fences/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_hazard_fences.m $(METAL_LDLIBS)

tests/.weight_tracking/quality.inc: tests/prepare_weight_tracking.py tests/test_native_prefill.m
	python3 tests/prepare_weight_tracking.py

tests/.weight_tracking/native_core.inc: tests/prepare_weight_tracking_native.py qw3_metal.m
	python3 tests/prepare_weight_tracking_native.py

qw3-weight-tracking-native-bench: tests/bench_weight_tracking_native.m tests/.weight_tracking/native_core.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_weight_tracking_native.m $(METAL_LDLIBS)

qw3-weight-tracking-bench: tests/bench_weight_tracking.m tests/weight_tracking_candidates.inc tests/.weight_tracking/quality.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_weight_tracking.m $(METAL_LDLIBS)

qw3-weight-tracking-test: tests/test_weight_tracking.m tests/weight_tracking_candidates.inc tests/.weight_tracking/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_weight_tracking.m $(METAL_LDLIBS)

tests/.hazard_tracking/quality.inc: tests/prepare_hazard_tracking.py tests/test_native_prefill.m tests/native_prefill_validation.inc qw3_metal.m
	python3 tests/prepare_hazard_tracking.py

qw3-hazard-tracking-test: tests/test_hazard_tracking.m tests/.hazard_tracking/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_hazard_tracking.m $(METAL_LDLIBS)

qw3-model-residency-bench: tests/bench_model_residency.m tests/model_residency_candidates.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_model_residency.m $(METAL_LDLIBS)

tests/.model_residency/quality.inc: tests/prepare_model_residency.py tests/test_native_prefill.m
	python3 tests/prepare_model_residency.py

qw3-model-residency-test: tests/test_model_residency.m tests/model_residency_candidates.inc tests/.model_residency/quality.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_model_residency.m $(METAL_LDLIBS)

qw3-gdn-cache-bench: tests/bench_gdn_cache.m tests/gdn_cache_candidates.inc tests/.gdn_cache/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gdn_cache.m $(METAL_LDLIBS)

tests/.gdn_windows/kernels.metal: tests/prepare_gdn_windows.py tests/test_native_prefill.m tests/test_gdn_cache.m tests/bench_iq3_groups.m metal/qw3_core_deltanet.metal
	python3 tests/prepare_gdn_windows.py

qw3-gdn-windows-test: tests/test_gdn_windows.m tests/gdn_windows_candidates.inc tests/.gdn_windows/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_windows.m $(METAL_LDLIBS)

qw3-gdn-windows-bench: tests/bench_gdn_windows.m tests/gdn_windows_candidates.inc tests/.gdn_windows/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gdn_windows.m $(METAL_LDLIBS)

tests/.gdn_local_gates/kernels.metal: tests/prepare_gdn_local_gates.py tests/test_native_prefill.m tests/test_gdn_cache.m tests/bench_iq3_groups.m metal/qw3_core_deltanet.metal
	python3 tests/prepare_gdn_local_gates.py

qw3-gdn-local-gates-test: tests/test_gdn_local_gates.m tests/gdn_local_gates_candidates.inc tests/.gdn_local_gates/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_local_gates.m $(METAL_LDLIBS)

qw3-gdn-local-gates-bench: tests/bench_gdn_local_gates.m tests/gdn_local_gates_candidates.inc tests/.gdn_local_gates/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gdn_local_gates.m $(METAL_LDLIBS)

tests/.gdn_two_columns/kernels.metal: tests/prepare_gdn_two_columns.py tests/test_native_prefill.m tests/test_gdn_cache.m tests/bench_iq3_groups.m metal/qw3_core_deltanet.metal qw3_metal.m
	python3 tests/prepare_gdn_two_columns.py

qw3-gdn-two-columns-test: tests/test_gdn_two_columns.m tests/gdn_two_columns_candidates.inc tests/.gdn_two_columns/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_gdn_two_columns.m $(METAL_LDLIBS)

qw3-gdn-two-columns-bench: tests/bench_gdn_two_columns.m tests/gdn_two_columns_candidates.inc tests/.gdn_two_columns/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_gdn_two_columns.m $(METAL_LDLIBS)

qw3-iq3-groups-test: tests/test_iq3_groups.m tests/iq3_groups_candidates.inc tests/.iq3_groups/wrapper.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_iq3_groups.m $(METAL_LDLIBS)

qw3-iq3-groups-bench: tests/bench_iq3_groups.m tests/iq3_groups_candidates.inc tests/.iq3_groups/wrapper.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_iq3_groups.m $(METAL_LDLIBS)

qw3-kstage-matrix-test: tests/test_kstage_matrix.m tests/kstage_candidates.inc tests/.kstage_candidates/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_kstage_matrix.m $(METAL_LDLIBS)

qw3-native-kstage-test: tests/test_native_kstage.m tests/native_prefill_validation.inc tests/.kstage_candidates/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_native_kstage.m $(METAL_LDLIBS)

qw3-kstage-prefill-test: tests/test_kstage_prefill.m tests/kstage_candidates.inc tests/.kstage_candidates/kernels.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_kstage_prefill.m $(METAL_LDLIBS)

qw3-kstage-prefill-bench: tests/bench_kstage_prefill.m tests/kstage_candidates.inc tests/.kstage_candidates/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_kstage_prefill.m $(METAL_LDLIBS)

qw3-kstage-paired-bench: tests/bench_kstage_paired.m tests/kstage_candidates.inc tests/.kstage_candidates/kernels.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_kstage_paired.m $(METAL_LDLIBS)

tests/.combination_candidates/core.inc: tests/prepare_combination_candidates.py tests/test_native_prefill.m qw3.c qw3_metal.m
	python3 tests/prepare_combination_candidates.py

tests/.sum8_candidates/candidates.inc: tests/prepare_sum8_candidates.py tests/sum8_dispatch.inc tests/combination_candidates.inc tests/.combination_candidates/core.inc metal/qw3_core_linear.metal qw3_metal.m
	python3 tests/prepare_sum8_candidates.py

qw3-sum8-test: tests/test_sum8_candidates.m tests/.sum8_candidates/candidates.inc tests/combination_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_sum8_candidates.m $(METAL_LDLIBS)

qw3-sum8-bench: tests/bench_sum8_candidates.m tests/.sum8_candidates/candidates.inc tests/combination_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_sum8_candidates.m $(METAL_LDLIBS)

qw3-combination-campaign: tests/bench_combination_campaign.m tests/.sum8_candidates/candidates.inc tests/combination_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_combination_campaign.m $(METAL_LDLIBS)

tests/.epilogue_candidates/candidates.inc: tests/prepare_epilogue_candidates.py tests/epilogue_dispatch.inc tests/.sum8_candidates/candidates.inc metal/qw3_core_moe.metal tests/bench_combination_campaign.m
	python3 tests/prepare_epilogue_candidates.py

tests/.register_iq4/kernel.metal: tests/prepare_register_iq4.py metal/qw3_core_moe.metal
	python3 tests/prepare_register_iq4.py

qw3-register-iq4-test: tests/test_register_iq4.m tests/register_iq4.inc tests/.register_iq4/kernel.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_register_iq4.m $(METAL_LDLIBS)

tests/.register_iq4_cached/kernel.metal: tests/prepare_register_iq4_cached.py tests/.register_iq4/kernel.metal tests/register_iq4.inc tests/test_register_iq4.m
	python3 tests/prepare_register_iq4_cached.py

qw3-register-iq4-cached-test: tests/test_register_iq4_cached.m tests/.register_iq4_cached/kernel.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_register_iq4_cached.m $(METAL_LDLIBS)

qw3-epilogue-test: tests/test_epilogue_candidates.m tests/.epilogue_candidates/candidates.inc tests/combination_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_epilogue_candidates.m $(METAL_LDLIBS)

qw3-epilogue-matrix-test: tests/test_epilogue_matrix.m tests/.epilogue_candidates/candidates.inc tests/combination_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_epilogue_matrix.m $(METAL_LDLIBS)

qw3-sum8-matrix-test: tests/test_sum8_matrix.m tests/.sum8_candidates/candidates.inc tests/combination_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_sum8_matrix.m $(METAL_LDLIBS)

qw3-epilogue-campaign: tests/.epilogue_candidates/candidates.inc tests/combination_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/.epilogue_candidates/bench.m $(METAL_LDLIBS)

qw3-combination-bench: tests/bench_combination_candidates.m tests/combination_candidates.inc tests/combination_candidates.metal tests/.combination_candidates/core.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_combination_candidates.m $(METAL_LDLIBS)

qw3-combination-test: tests/test_combination_candidates.m tests/combination_candidates.inc tests/combination_candidates.metal tests/.combination_candidates/core.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_combination_candidates.m $(METAL_LDLIBS)

tests/.micro_candidates/core.inc: tests/prepare_micro_candidates.py tests/test_native_prefill.m qw3.c
	python3 tests/prepare_micro_candidates.py

qw3-micro-bench: tests/bench_micro_candidates.m tests/micro_candidates.inc tests/micro_candidates.metal tests/.micro_candidates/core.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/bench_micro_candidates.m $(METAL_LDLIBS)

qw3-micro-test: tests/test_micro_candidates.m tests/test_native_prefill.m tests/micro_candidates.inc tests/micro_candidates.metal tests/.micro_candidates/core.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_micro_candidates.m $(METAL_LDLIBS)

qw3-micro-q8-test: tests/test_micro_q8.m tests/micro_candidates.inc tests/micro_candidates.metal tests/.micro_candidates/core.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_micro_q8.m $(METAL_LDLIBS)

qw3-copy-control-bench: tests/bench_copy_control.m tests/native_prefill_validation.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_copy_control.m $(METAL_LDLIBS)

qw3-copy-control-test: tests/test_copy_control.m tests/test_native_prefill.m tests/native_prefill_validation.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_copy_control.m $(METAL_LDLIBS)

qw3-revisit-prefill-bench: tests/bench_revisit_prefill.m tests/bench_projection_prefill.m tests/projection_candidates.inc tests/projection_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_revisit_prefill.m $(METAL_LDLIBS)

qw3-revisit-prefill-test: tests/test_revisit_prefill.m tests/test_projection_prefill.m tests/projection_candidates.inc tests/projection_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_revisit_prefill.m $(METAL_LDLIBS)

qw3-native-prefill-test: tests/test_native_prefill.m tests/native_prefill_validation.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_native_prefill.m $(METAL_LDLIBS)

qw3-native-prefill-matmul-test: tests/test_native_prefill_matmul.m tests/native_prefill_validation.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_native_prefill_matmul.m $(METAL_LDLIBS)

qw3-packed8-scratch-bench: tests/bench_packed8_scratch.m tests/bench_scratch_prefill.m tests/scratch_candidates.inc tests/scratch_matmul.inc tests/gdn_prefill_dispatch.inc tests/gdn_prefill_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_packed8_scratch.m $(METAL_LDLIBS)

qw3-packed8-scratch-test: tests/test_packed8_scratch.m tests/test_scratch_prefill.m tests/scratch_candidates.inc tests/scratch_matmul.inc tests/gdn_prefill_dispatch.inc tests/gdn_prefill_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_packed8_scratch.m $(METAL_LDLIBS)

qw3-gdn-prefill-bench: tests/bench_gdn_prefill.m tests/gdn_prefill_candidates.inc tests/gdn_prefill_dispatch.inc tests/gdn_prefill_candidates.metal qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_gdn_prefill.m $(METAL_LDLIBS)

qw3-gdn-prefill-test: tests/test_gdn_prefill.m tests/gdn_prefill_candidates.inc tests/gdn_prefill_dispatch.inc tests/gdn_prefill_candidates.metal qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_gdn_prefill.m $(METAL_LDLIBS)

qw3-dense-candidates-bench: tests/bench_dense_candidates.m tests/dense_candidates.inc qw3_bench.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_dense_candidates.m $(METAL_LDLIBS)

qw3-dense-candidates-test: tests/test_dense_candidates.m tests/dense_candidates.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/test_dense_candidates.m $(METAL_LDLIBS)

.PHONY: test-decode-trials-quant
test-decode-trials-quant: qw3-decode-trials-bench
	sh tests/test_decode_trials_quant.sh

qw3-prefill-staging-bench: tests/bench_prefill_staging.m tests/prefill_copy_probe.inc qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_prefill_staging.m $(METAL_LDLIBS)

test-iq3-stream-quant: qw3-iq3-stream-bench
	sh tests/test_iq3_stream.sh

.PHONY: test-prefill-copy
test-prefill-copy: qw3-prefill-staging-bench
	sh tests/test_prefill_copy.sh

qw3-block-cost-bench: tests/bench_verify_blocks.c qw3.c qw3.h qw3_metal.h qw3_runtime_fingerprint.h qw3_metal.o
	$(CC) $(CFLAGS) -fno-finite-math-only -o $@ tests/bench_verify_blocks.c qw3_metal.o $(METAL_LDLIBS)

qw3-short-batch-bench: tests/bench_short_batch.m tests/bench_verify_blocks.c qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -o $@ tests/bench_short_batch.m $(METAL_LDLIBS)

qw3-dense-metal-logits-test: tests/test_dense_logits.c qw3.h qw3_metal_core.o qw3_metal.o
	$(CC) $(filter-out -std=c99,$(CFLAGS)) -fno-finite-math-only -std=c11 \
		-I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include \
		-o $@ tests/test_dense_logits.c qw3_metal_core.o qw3_metal.o \
		-L$(GGML_LIB_DIR) -Wl,-rpath,$(abspath $(GGML_LIB_DIR)) \
		-lllama -lggml-cpu -lggml-base -lggml $(METAL_LDLIBS)

qw3-rope-metal-test: tests/test_rope.c qw3.c qw3.h qw3_metal.h qw3_runtime_fingerprint.h qw3_metal.o
	$(CC) $(CFLAGS) -fno-finite-math-only -o $@ tests/test_rope.c qw3_metal.o $(METAL_LDLIBS)

qw3-agent: qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o $(METAL_LDLIBS)

qw3-agent-context-pressure-test: tests/test_agent_context_pressure.m qw3_agent.c qw3_agent_unicode.h qw3.c qw3.h qw3_metal.m qw3_metal.h $(METAL_SRCS) qw3_runtime_fingerprint.h linenoise_qw3.o
	$(CC) $(OBJCFLAGS) -fno-finite-math-only -I. -o $@ tests/test_agent_context_pressure.m linenoise_qw3.o $(METAL_LDLIBS)

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

qw3-agent-context-pressure-test:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-eval: qw3-eval-cpu
	cp qw3-eval-cpu qw3-eval

qw3-metal qw3-bench-metal qw3-eval-metal qw3-sampled-decode-bench qw3-decode-ab-bench qw3-dense-metal-logits-test qw3-rope-metal-test:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-q4-words-test qw3-q4-words-model-test qw3-q4-words-native-bench qw3-q4-rows-test qw3-q4-rows-model-test qw3-q4-rows-native-bench qw3-q4-narrow-test qw3-q4-narrow-model-test qw3-q4-narrow-native-bench qw3-iq3-input-test qw3-iq3-input-model-test qw3-iq3-input-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-shape-test qw3-iq3-shape-model-test qw3-iq3-shape-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-q5-words-test qw3-q5-words-model-test qw3-q5-words-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-add-test qw3-iq3-add-model-test qw3-iq3-add-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-pair-grid-test qw3-iq3-pair-grid-model-test qw3-iq3-pair-grid-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-immutable-test qw3-iq3-immutable-model-test qw3-iq3-immutable-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-rows-test qw3-iq3-rows-model-test qw3-iq3-rows-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-share-test qw3-iq3-share-model-test qw3-iq3-share-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-gqa-barrier-test qw3-gqa-barrier-model-test qw3-gqa-barrier-native-bench qw3-iq4-shape-test qw3-iq4-shape-model-test qw3-iq4-shape-native-bench qw3-iq4-simd-lut-test qw3-iq4-simd-lut-model-test qw3-iq4-simd-lut-native-bench qw3-iq4-unroll-test qw3-iq4-unroll-model-test qw3-iq4-unroll-native-bench qw3-iq4-share-test qw3-iq4-share-model-test qw3-iq4-share-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-gdn-two128-test qw3-gdn-two128-model-test qw3-gdn-two128-native-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-iq3-hint-test qw3-iq3-hint-model-test qw3-iq3-hint-native-bench qw3-iq3-words-test qw3-iq3-words-model-test qw3-iq3-words-native-bench qw3-iq3-grid-test qw3-iq3-grid-model-test qw3-iq3-grid-native-bench qw3-iq3-private-native-bench qw3-iq3-private-test qw3-iq3-private-bench qw3-iq3-layout-test qw3-iq3-layout-bench qw3-iq4-packed-lut-test qw3-iq4-packed-lut-bench qw3-iq3-signed-test qw3-iq3-signed-bench qw3-iq4-rows-test qw3-iq4-rows-bench qw3-iq4-rows-cost-bench qw3-iq4-loads-test qw3-iq4-loads-bench qw3-decode-profile-test qw3-hazard-fences-native-bench qw3-hazard-fences-test qw3-weight-tracking-native-bench qw3-weight-tracking-bench qw3-weight-tracking-test qw3-hazard-tracking-test qw3-gdn-two-columns-test qw3-gdn-two-columns-bench qw3-gdn-local-gates-test qw3-gdn-local-gates-bench qw3-gdn-windows-test qw3-gdn-windows-bench qw3-model-residency-test qw3-model-residency-bench qw3-native-gdn-cache-test qw3-gdn-cache-test qw3-gdn-cache-bench qw3-iq3-groups-test qw3-iq3-groups-bench qw3-native-kstage-test qw3-kstage-paired-bench qw3-kstage-matrix-test qw3-kstage-prefill-test qw3-kstage-prefill-bench qw3-register-iq4-cached-test qw3-register-iq4-test qw3-epilogue-matrix-test qw3-sum8-matrix-test qw3-epilogue-test qw3-epilogue-campaign qw3-combination-campaign qw3-combination-bench qw3-combination-test qw3-sum8-bench qw3-sum8-test:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1

qw3-micro-bench qw3-micro-test qw3-micro-q8-test qw3-copy-control-bench qw3-copy-control-test qw3-revisit-prefill-bench qw3-revisit-prefill-test qw3-native-prefill-test qw3-native-prefill-matmul-test qw3-packed8-scratch-bench qw3-packed8-scratch-test qw3-scratch-prefill-bench qw3-scratch-prefill-test qw3-scratch-matmul-test qw3-projection-cost-bench qw3-projection-prefill-bench qw3-projection-prefill-test qw3-gdn-prefill-bench qw3-gdn-prefill-test qw3-q5-head-bench qw3-iq3-stream-bench qw3-decode-trials-bench qw3-dense-candidates-bench qw3-dense-candidates-test qw3-prefill-staging-bench qw3-block-cost-bench qw3-short-batch-bench:
	@echo "$@ requires Darwin/Apple Metal"
	@exit 1
endif

tools codenav:
	$(MAKE) -C codenavsrc

bench-sampled-decode: qw3-sampled-decode-bench
	./qw3-sampled-decode-bench "$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"

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
	$(CC) $(filter-out -std=c99,$(CFLAGS)) -fno-finite-math-only -std=c11 \
		-I$(LLAMA_CPP)/include -I$(LLAMA_CPP)/ggml/include \
		-o $@ tests/test_dense_logits.c qw3_cpu_core.o \
		-L$(GGML_LIB_DIR) -Wl,-rpath,$(abspath $(GGML_LIB_DIR)) \
		-lllama -lggml-cpu -lggml-base -lggml $(LDLIBS)

test-dense-logits: qw3-dense-logits-test
	./qw3-dense-logits-test \
		$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf} 66

qw3-rope-cpu-test: tests/test_rope.c qw3.c qw3.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DQW3_NO_METAL -o $@ tests/test_rope.c $(LDLIBS)

test-rope: qw3-rope-cpu-test
	./qw3-rope-cpu-test

test-rope-metal: qw3-rope-metal-test
	./qw3-rope-metal-test

test-dense-metal-logits: qw3-dense-metal-logits-test
	QW3_DENSE_METAL=1 ./qw3-dense-metal-logits-test \
		$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf} 66 30057 14578 1789
	QW3_DENSE_METAL=1 ./qw3-dense-metal-logits-test \
		$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf} 66 561 1118 321 2018 41131 314 279

test-short-batch-quant: qw3-short-batch-bench
	QW3_BLOCK_QUANT_ONLY=1 ./qw3-short-batch-bench \
		$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}

test-short-batch: qw3-short-batch-bench
	sh tests/test_short_batch.sh

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

test-agent-python-smoke: qw3-agent
	sh tests/test_agent_python_smoke.sh

test-agent-git-smoke: qw3-agent
	sh tests/test_agent_git_smoke.sh

test-agent-prefix-cache: qw3-agent
	sh tests/test_agent_prefix_cache.sh

test-agent-color: qw3-agent
	QW3_AGENT_COLOR_SELFTEST=1 ./qw3-agent

test-agent-context: qw3-agent
	QW3_AGENT_CONTEXT_LEDGER_SELFTEST=1 ./qw3-agent

test-agent-context-pressure: qw3-agent-context-pressure-test
	./qw3-agent-context-pressure-test $${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf}

test-agent-web-page: qw3-agent
	python3 tests/test_agent_web_page.py

test-agent-store: qw3_cpu_core.o linenoise_qw3.o
	@set -e; probe=$$(mktemp "$${TMPDIR:-/tmp}/qw3-agent-store.XXXXXX"); \
	trap 'rm -f "$$probe"' EXIT; \
	$(CC) $(CFLAGS) -DQW3_NO_METAL -o "$$probe" tests/test_agent_store.c \
		qw3_cpu_core.o linenoise_qw3.o $(LDLIBS); \
	"$$probe"

test-agent-llama-reference: qw3-agent
	python3 tests/test_agent_llama_reference.py

test-agent-web-moe: qw3-agent
	python3 tests/test_agent_llama_reference.py --candidate-only \
		--model $${QW3_MOE_MODEL:-../../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}

test-agent-web-think: qw3-agent
	python3 tests/test_agent_llama_reference.py --candidate-only --think

test-agent-repeat: qw3-agent
	QW3_AGENT_REPEAT_GUARD_SELFTEST=1 ./qw3-agent

test-agent-repeat-format: qw3-agent
	python3 tests/test_agent_repeat_format.py

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

test-dense-gqa-decode: qw3-test
	QW3_METAL_BIN=./qw3-test \
		QW3_MODEL=$${QW3_DENSE_MODEL:-../../models/Qwen3.8-27B-UD-Q3_K_XL.gguf} \
		QW3_METAL_GQA_FLASH_DECODE_GROUPED=$${QW3_METAL_GQA_FLASH_DECODE_GROUPED:-1} \
		QW3_GQA_DECODE_CONTEXTS="$${QW3_GQA_DECODE_CONTEXTS:-1023 1024 1025 4097 16385}" \
		sh tests/test_metal_gqa_decode.sh

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
	rm -f qw3-q4-words-test qw3-q4-words-model-test qw3-q4-words-native-bench
	rm -f qw3-q4-rows-test qw3-q4-rows-model-test qw3-q4-rows-native-bench
	rm -f qw3-q4-narrow-test qw3-q4-narrow-model-test qw3-q4-narrow-native-bench
	rm -f qw3-iq3-input-test qw3-iq3-input-model-test qw3-iq3-input-native-bench
	rm -f qw3-iq3-shape-test qw3-iq3-shape-model-test qw3-iq3-shape-native-bench
	rm -f qw3-q5-words-test qw3-q5-words-model-test qw3-q5-words-native-bench
	rm -f qw3-iq3-add-test qw3-iq3-add-model-test qw3-iq3-add-native-bench
	rm -f qw3-iq3-pair-grid-test qw3-iq3-pair-grid-model-test qw3-iq3-pair-grid-native-bench
	rm -f qw3-gqa-barrier-test qw3-gqa-barrier-model-test qw3-gqa-barrier-native-bench
	rm -f qw3-iq4-shape-test qw3-iq4-shape-model-test qw3-iq4-shape-native-bench
	rm -f qw3-iq4-simd-lut-test qw3-iq4-simd-lut-model-test qw3-iq4-simd-lut-native-bench
	rm -f qw3-iq4-unroll-test qw3-iq4-unroll-model-test qw3-iq4-unroll-native-bench
	rm -f qw3-agent-context-pressure-test
	rm -f qw3-gdn-two-columns-test qw3-gdn-two-columns-bench
	rm -f qw3-gdn-two128-test qw3-gdn-two128-model-test qw3-gdn-two128-native-bench
	rm -f qw3-gdn-local-gates-test qw3-gdn-local-gates-bench
	rm -f qw3-gdn-windows-test qw3-gdn-windows-bench
	rm -f qw3-model-residency-bench
	rm -f qw3-iq3-grid-test qw3-iq3-grid-model-test qw3-iq3-grid-native-bench
	rm -f qw3-iq3-words-test qw3-iq3-words-model-test qw3-iq3-words-native-bench
	rm -f qw3-iq3-hint-test qw3-iq3-hint-model-test qw3-iq3-hint-native-bench
	rm -f qw3-iq3-immutable-test qw3-iq3-immutable-model-test qw3-iq3-immutable-native-bench
	rm -f qw3-iq3-rows-test qw3-iq3-rows-model-test qw3-iq3-rows-native-bench
	rm -f qw3-iq3-share-test qw3-iq3-share-model-test qw3-iq3-share-native-bench
	rm -f qw3-iq4-share-test qw3-iq4-share-model-test qw3-iq4-share-native-bench
	rm -f qw3-iq3-signed-test qw3-iq3-signed-bench
	rm -f qw3-iq3-layout-test qw3-iq3-layout-bench
	rm -f qw3-iq3-private-test qw3-iq3-private-bench qw3-iq3-private-native-bench
	rm -f qw3-iq4-packed-lut-test qw3-iq4-packed-lut-bench
	rm -f qw3-iq4-rows-test qw3-iq4-rows-bench qw3-iq4-rows-cost-bench
	rm -f qw3-iq4-loads-test qw3-iq4-loads-bench
	rm -f qw3-model-residency-test qw3-decode-profile-test qw3-hazard-fences-test qw3-hazard-fences-native-bench qw3-hazard-tracking-test qw3-weight-tracking-test qw3-weight-tracking-bench qw3-weight-tracking-native-bench
	rm -f qw3-native-gdn-cache-test
	rm -f qw3-gdn-cache-test qw3-gdn-cache-bench
	rm -f qw3-iq3-groups-test qw3-iq3-groups-bench
	rm -f qw3-kstage-matrix-test qw3-kstage-prefill-test qw3-kstage-prefill-bench qw3-kstage-paired-bench
	rm -f qw3-native-kstage-test
	rm -f qw3-register-iq4-test
	rm -f qw3-register-iq4-cached-test
	rm -f qw3-epilogue-matrix-test qw3-sum8-matrix-test
	rm -f qw3-epilogue-test qw3-epilogue-campaign qw3-combination-campaign qw3-combination-bench qw3-combination-test qw3-sum8-bench qw3-sum8-test
	rm -f qw3_runtime_fingerprint.h
	rm -f qw3-scratch-prefill-bench qw3-scratch-prefill-test qw3-scratch-matmul-test
	rm -f qw3-packed8-scratch-bench qw3-packed8-scratch-test
	rm -f qw3-micro-bench qw3-micro-test qw3-micro-q8-test qw3-copy-control-bench qw3-copy-control-test qw3-revisit-prefill-bench qw3-revisit-prefill-test qw3-native-prefill-test qw3-native-prefill-matmul-test qw3 qw3-cli qw3-cpu qw3-test qw3-cpu-test qw3-quant-k-test qw3-dense-logits-test qw3-metal qw3-agent qw3-agent-cpu \
		qw3-bench qw3-bench-cpu qw3-bench-metal qw3-sampled-decode-bench qw3-decode-ab-bench \
		qw3-projection-cost-bench qw3-projection-prefill-bench qw3-projection-prefill-test qw3-gdn-prefill-bench qw3-gdn-prefill-test qw3-q5-head-bench qw3-iq3-stream-bench qw3-decode-trials-bench qw3-dense-candidates-bench qw3-dense-candidates-test qw3-prefill-staging-bench qw3-block-cost-bench qw3-short-batch-bench \
		qw3-dense-metal-logits-test qw3-rope-cpu-test qw3-rope-metal-test \
		qw3-eval qw3-eval-cpu qw3-eval-metal *.o
