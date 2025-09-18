# Compiler and flags
CC = gcc
# OpenMP Flag
ifdef USE_OPENMP
    OPENMP_FLAG = -fopenmp
else
    OPENMP_FLAG = 
endif

CFLAGS = -I nn/include -I tests -Wall -Wextra -Werror -Wpedantic -Wstrict-prototypes -Wold-style-definition -g $(CU_CFLAGS) $(OPENMP_FLAG)

# Source files
SRCS = $(shell find nn/src -name '*.c' -not -path 'nn/src/main.c')
TEST_SRCS = tests/core_cunit.c tests/nn_cunit.c tests/cunit_runner.c tests/test_utils.c

# Object files
OBJS = $(SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Executables
EXAMPLE_SRCS = $(wildcard nn/src/examples/*.c)
EXAMPLE_OBJS = $(EXAMPLE_SRCS:.c=.o)
EXAMPLE_EXECS = $(EXAMPLE_SRCS:.c=)
TEST_RUNNER = tests/cunit_runner_exec

# CUnit flags
CU_CFLAGS = $(shell pkg-config --cflags cunit)
CU_LIBS = $(shell pkg-config --libs cunit)

# Targets
all: $(EXAMPLE_EXECS) $(TEST_RUNNER)

$(EXAMPLE_EXECS): nn/src/examples/%: $(filter-out nn/src/examples/%.o, $(OBJS)) nn/src/examples/%.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TEST_RUNNER): $(filter-out $(EXAMPLE_OBJS), $(OBJS)) $(TEST_OBJS)
	$(CC) $(CFLAGS) $(CU_CFLAGS) -o $@ $^ $(CU_LIBS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TEST_OBJS) $(EXAMPLE_OBJS) $(EXAMPLE_EXECS) $(TEST_RUNNER)

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

format:
	clang-format -i $(SRCS) $(TEST_SRCS) nn/include/*.h

.PHONY: all clean test format