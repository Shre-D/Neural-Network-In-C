#!/bin/sh

# Format file when I make a commit.
find . -name "*.c" -o -name "*.h" | xargs clang-format -i --style=Google
