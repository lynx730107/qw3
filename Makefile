CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -pthread

UNAME_S := $(shell uname -s)
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
endif

.PHONY: all clean qw3 agent test-vectors test-metal-smoke test-metal-logits test-metal-logits-concurrent metal qw3-eval qw3-eval-metal

all: qw3-cpu qw3-agent qw3-eval qw3-bench
ifeq ($(UNAME_S),Darwin)
all: qw3-metal qw3-eval-metal qw3-bench-metal
endif
qw3: qw3-cpu
metal: qw3-metal
agent: qw3-agent

qw3.o: qw3.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3.c

qw3_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_agent.o: qw3_agent.c qw3.h ../linenoise.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_agent.c

qw3_bench.o: qw3_bench.c qw3.h
	$(CC) $(CFLAGS)  -c -o $@ qw3_bench.c

linenoise_qw3.o: ../linenoise.c ../linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ../linenoise.c

qw3-cpu: qw3_cli.o qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_cli.o qw3.o $(LDLIBS)

qw3-bench: qw3_bench.o qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3.o $(LDLIBS)

qw3-agent-cpu: qw3_agent.o qw3.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_agent.o qw3.o linenoise_qw3.o $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
qw3_metal_core.o: qw3.c qw3.h qw3_metal.h
	$(CC) $(CFLAGS) -c -o $@ qw3.c

qw3_metal_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_CLI_ENABLE_INTERNAL_TESTS=1 -c -o $@ qw3_cli.c

qw3_metal_agent.o: qw3_agent.c qw3.h ../linenoise.h
	$(CC) $(CFLAGS) -c -o $@ qw3_agent.c

qw3_metal.o: qw3_metal.m qw3_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ qw3_metal.m

qw3-metal: qw3_metal_cli.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-bench-metal: qw3_bench.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_bench.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)

qw3-agent: qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_agent.o qw3_metal_core.o qw3_metal.o linenoise_qw3.o $(METAL_LDLIBS)
else
qw3-metal:
	@echo "qw3-metal requires Darwin/Metal"
	@exit 1

qw3-agent: qw3-agent-cpu
endif

qw3_eval.o: qw3_eval.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_eval.c

qw3-eval: qw3_eval.o qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_eval.o qw3.o $(LDLIBS)

test-vectors: qw3-cpu
	sh tests/test_vectors.sh

test-metal-smoke: qw3-metal
	sh tests/test_metal_smoke.sh

test-metal-logits: qw3-metal
	sh tests/test_metal_logits_regression.sh

test-metal-logits-concurrent: qw3-metal
	QW3_METAL_PREFILL_CONCURRENT=1 sh tests/test_metal_logits_regression.sh

ifeq ($(UNAME_S),Darwin)
qw3_eval_metal.o: qw3_eval.c qw3.h qw3_metal.h
	$(CC) $(CFLAGS) -c -o $@ qw3_eval.c

qw3-eval-metal: qw3_eval_metal.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_eval_metal.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)
endif

clean:
	rm -f qw3-cpu qw3-metal qw3-agent qw3-agent-cpu qw3-eval qw3-eval-metal *.o
