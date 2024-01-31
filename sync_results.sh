#!/usr/bin/env bash

TARGET_DIR=${TARGET_DIR:-$(cat TARGET_DIR)}
DIRECTION=${1:-"to-target"}

if [ $DIRECTION == "to-target" ]; then
    echo "Syncing results to $TARGET_DIR..."
    rsync -r -t -P --delete ./results "$TARGET_DIR"

    echo
    echo "Syncing saved_experiments to $TARGET_DIR..."
    rsync -r -t -P --delete ./saved_experiments "$TARGET_DIR"
elif [ $DIRECTION == "from-target" ]; then
    echo "Syncing results from $TARGET_DIR..."
    rsync -r -t -P --delete "$TARGET_DIR/results" .

    echo
    echo "Syncing saved_experiments from $TARGET_DIR..."
    rsync -r -t -P --delete "$TARGET_DIR/saved_experiments" .
else
    echo "Unknown direction: $DIRECTION"
    exit 1
fi

echo "Done."
