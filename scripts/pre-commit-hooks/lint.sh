#!/bin/sh

# This script runs Cppcheck on the entire project to find potential bugs and
# violations of coding standards.
#
# The '--enable=all' flag turns on all checks.
# The '--inconclusive' flag reports warnings that may or may not be valid.
# The '--error-exitcode=1' ensures the script fails on error

# I'd like the checking to be as strict as possible because C is so good.
# I actually know everything happening at the lower level
# And I don't have to worry about subpar warnings

cppcheck --enable=all -I nn/include --suppress=missingIncludeSystem --suppress=unusedFunction --inconclusive --error-exitcode=1 .