CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -pthread

UNAME_S := $(shell uname -s)
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
endif

.PHONY: all clean qw3 test-vectors test-metal-smoke metal

all: qw3-cpu
qw3: qw3-cpu
metal: qw3-metal

qw3.o: qw3.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3.c

qw3_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -DQW3_NO_METAL -c -o $@ qw3_cli.c

qw3-cpu: qw3_cli.o qw3.o
	$(CC) $(CFLAGS) -o $@ qw3_cli.o qw3.o $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
qw3_metal_core.o: qw3.c qw3.h qw3_metal.h
	$(CC) $(CFLAGS) -c -o $@ qw3.c

qw3_metal_cli.o: qw3_cli.c qw3.h
	$(CC) $(CFLAGS) -c -o $@ qw3_cli.c

qw3_metal.o: qw3_metal.m qw3_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ qw3_metal.m

qw3-metal: qw3_metal_cli.o qw3_metal_core.o qw3_metal.o
	$(CC) $(CFLAGS) -o $@ qw3_metal_cli.o qw3_metal_core.o qw3_metal.o $(METAL_LDLIBS)
else
qw3-metal:
	@echo "qw3-metal requires Darwin/Metal"
	@exit 1
endif

test-vectors: qw3-cpu
	sh tests/test_vectors.sh

test-metal-smoke: qw3-metal
	sh tests/test_metal_smoke.sh

clean:
	rm -f qw3-cpu qw3-metal *.o
