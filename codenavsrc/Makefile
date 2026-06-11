CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
INCLUDES = -Ithird_party/tree-sitter/lib/include -Ithird_party/tree-sitter/lib/src
GIT = git

TARGET = codenav
SOURCES = main.c code_navigation.c \
	third_party/tree-sitter/lib/src/lib.c \
	third_party/tree-sitter-c/src/parser.c \
	third_party/tree-sitter-python/src/parser.c \
	third_party/tree-sitter-python/src/scanner.c \
	third_party/tree-sitter-cpp/src/parser.c \
	third_party/tree-sitter-cpp/src/scanner.c \
	third_party/tree-sitter-javascript/src/parser.c \
	third_party/tree-sitter-javascript/src/scanner.c \
	third_party/tree-sitter-go/src/parser.c \
	third_party/tree-sitter-c-sharp/src/parser.c \
	third_party/tree-sitter-c-sharp/src/scanner.c \
	third_party/tree-sitter-ruby/src/parser.c \
	third_party/tree-sitter-ruby/src/scanner.c \
	third_party/tree-sitter-objc/src/parser.c

TREE_SITTER_REPOS = \
	third_party/tree-sitter \
	third_party/tree-sitter-c \
	third_party/tree-sitter-python \
	third_party/tree-sitter-cpp \
	third_party/tree-sitter-javascript \
	third_party/tree-sitter-go \
	third_party/tree-sitter-c-sharp \
	third_party/tree-sitter-ruby \
	third_party/tree-sitter-objc

.PHONY: all deps clean rebuild

all: deps $(TARGET)

$(TARGET): $(SOURCES) code_navigation.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES)

deps: $(TREE_SITTER_REPOS)

third_party:
	mkdir -p third_party

third_party/tree-sitter: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter third_party/tree-sitter

third_party/tree-sitter-c: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-c third_party/tree-sitter-c

third_party/tree-sitter-python: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-python third_party/tree-sitter-python

third_party/tree-sitter-cpp: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-cpp third_party/tree-sitter-cpp

third_party/tree-sitter-javascript: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-javascript third_party/tree-sitter-javascript

third_party/tree-sitter-go: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-go third_party/tree-sitter-go

third_party/tree-sitter-c-sharp: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-c-sharp third_party/tree-sitter-c-sharp

third_party/tree-sitter-ruby: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter/tree-sitter-ruby third_party/tree-sitter-ruby

third_party/tree-sitter-objc: | third_party
	$(GIT) clone --depth 1 https://github.com/tree-sitter-grammars/tree-sitter-objc third_party/tree-sitter-objc

clean:
	$(RM) $(TARGET)

rebuild: clean all
