#!/bin/sh

# Make all scripts inside the scripts folder executable in one go
# You just need to run ```chmod +x ./scripts/init.sh``` and execute it to give
# permissions to all files at once.
find . -type f -name "*.sh" -exec chmod +x {} +
