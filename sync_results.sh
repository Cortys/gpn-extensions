#!/usr/bin/env bash

TARGET_DIR=${TARGET_DIR:-$(cat TARGET_DIR)}

echo "Syncing results to $TARGET_DIR..."
rsync -r -t -P --delete ./results "$TARGET_DIR"

echo
echo "Syncing saved_experiments to $TARGET_DIR..."
rsync -r -t -P --delete ./saved_experiments "$TARGET_DIR"

echo "Done."
