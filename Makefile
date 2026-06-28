CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -pthread

UNAME_S := $(shell uname -s)
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
endif

.PHONY: all clean cpu metal agent tools codenav hotlist code-profile-dataset \
		code-profile-c-dataset ssd-streaming-bench \
	test-vectors test-metal-smoke test-metal-logits test-metal-logits-concurrent \
	qw3-metal qw3-bench-metal qw3-eval-metal

all: qw3-cli qw3-agent qw3-eval qw3-bench
agent: qw3-agent
cpu: qw3-cpu qw3-agent-cpu qw3-eval-cpu qw3-bench-cpu
HOTLIST_TOP ?= 4096
CODE_PROFILE_DATASET ?= humaneval-x-cpp
CODE_PROFILE_TASKS ?= 20
CODE_PROFILE_MODE ?= mixed

qw3_ssd.o: qw3_ssd.c qw3_ssd.h
	$(CC) $(CFLAGS) -c -o $@ qw3_ssd.c

qw3_cpu_core.o: qw3.c qw3.h qw3_ssd.h qw3_streaming_hotlist.inc
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3.c

qw3_cpu_cli.o: qw3_cli.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_cli.c

qw3_cpu_cli_test.o: qw3_cli.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_cpu_agent.o: qw3_agent.c qw3.h qw3_ssd.h linenoise.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_agent.c

qw3_eval_cpu.o: qw3_eval.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_eval.c

qw3_bench.o: qw3_bench.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -c -o $@ qw3_bench.c

linenoise_qw3.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

qw3-cpu: qw3_cpu_cli.o qw3_cpu_core.o qw3_ssd.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_cli.o qw3_cpu_core.o qw3_ssd.o $(LDLIBS)

qw3-cpu-test: qw3_cpu_cli_test.o qw3_cpu_core.o qw3_ssd.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_cli_test.o qw3_cpu_core.o qw3_ssd.o $(LDLIBS)

qw3-bench-cpu: qw3_bench.o qw3_cpu_core.o qw3_ssd.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3_cpu_core.o qw3_ssd.o $(LDLIBS)

qw3-agent-cpu: qw3_cpu_agent.o qw3_cpu_core.o qw3_ssd.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_cpu_agent.o qw3_cpu_core.o qw3_ssd.o linenoise_qw3.o $(LDLIBS)

qw3-eval-cpu: qw3_eval_cpu.o qw3_cpu_core.o qw3_ssd.o
	$(CC) $(CFLAGS) -o $@ qw3_eval_cpu.o qw3_cpu_core.o qw3_ssd.o $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
metal: qw3-cli qw3-agent qw3-eval qw3-bench

qw3_metal_core.o: qw3.c qw3.h qw3_ssd.h qw3_metal.h qw3_streaming_hotlist.inc
	$(CC) $(CFLAGS) -c -o $@ qw3.c

qw3_metal_cli.o: qw3_cli.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -c -o $@ qw3_cli.c

qw3_metal_cli_test.o: qw3_cli.c qw3.h qw3_ssd.h
	$(CC) $(CFLAGS) -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_metal_agent.o: qw3_agent.c qw3.h qw3_ssd.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ qw3_agent.c

qw3_eval_metal.o: qw3_eval.c qw3.h qw3_ssd.h qw3_metal.h
	$(CC) $(CFLAGS) -c -o $@ qw3_eval.c

qw3_metal.o: qw3_metal.m qw3_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ qw3_metal.m

qw3-cli: qw3_metal_cli.o qw3_metal_core.o qw3_ssd.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli.o qw3_metal_core.o qw3_ssd.o qw3_metal.o $(METAL_LDLIBS)

qw3-test: qw3_metal_cli_test.o qw3_metal_core.o qw3_ssd.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli_test.o qw3_metal_core.o qw3_ssd.o qw3_metal.o $(METAL_LDLIBS)

qw3-bench: qw3_bench.o qw3_metal_core.o qw3_ssd.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3_metal_core.o qw3_ssd.o qw3_metal.o $(METAL_LDLIBS)

qw3-agent: qw3_metal_agent.o qw3_metal_core.o qw3_ssd.o qw3_metal.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_agent.o qw3_metal_core.o qw3_ssd.o qw3_metal.o linenoise_qw3.o $(METAL_LDLIBS)

qw3-eval: qw3_eval_metal.o qw3_metal_core.o qw3_ssd.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_eval_metal.o qw3_metal_core.o qw3_ssd.o qw3_metal.o $(METAL_LDLIBS)

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

hotlist:
	@if [ -z "$(PROFILE)" ]; then \
		echo "usage: make hotlist PROFILE=profiles/code.tsv [HOTLIST_TOP=4096]"; \
		exit 2; \
	fi
	python3 scripts/qw3_profile_to_hotlist_inc.py --top $(HOTLIST_TOP) \
		--output qw3_streaming_hotlist.inc $(PROFILE)

code-profile-dataset:
	python3 scripts/download_code_profile_dataset.py \
		--dataset $(CODE_PROFILE_DATASET) \
		--mode $(CODE_PROFILE_MODE) --max-tasks $(CODE_PROFILE_TASKS)

code-profile-c-dataset:
	python3 scripts/download_code_profile_dataset.py \
		--dataset synthetic-c --out-dir datasets/synthetic-c \
		--mode $(CODE_PROFILE_MODE) --max-tasks $(CODE_PROFILE_TASKS)

ssd-streaming-bench: qw3-cli
	sh scripts/bench_ssd_streaming.sh

test-vectors: qw3-cpu
	sh tests/test_vectors.sh

test-metal-smoke: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_smoke.sh

test-metal-logits: qw3-test
	QW3_METAL_BIN=./qw3-test sh tests/test_metal_logits_regression.sh

test-metal-logits-concurrent: qw3-test
	QW3_METAL_BIN=./qw3-test QW3_METAL_PREFILL_CONCURRENT=1 sh tests/test_metal_logits_regression.sh

clean:
	rm -f qw3 qw3-cpu qw3-test qw3-cpu-test qw3-metal qw3-agent qw3-agent-cpu \
		qw3-cli \
		qw3-bench qw3-bench-cpu qw3-bench-metal \
		qw3-eval qw3-eval-cpu qw3-eval-metal *.o
