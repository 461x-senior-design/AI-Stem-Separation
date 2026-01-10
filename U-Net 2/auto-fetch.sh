#!/bin/bash
while true; do
  git fetch origin claude/initial-setup-011CUpNZgwYnxz6wJsGE4hFb
  echo "Fetched at $(date)"
  sleep 60  # Check every 60 seconds (adjust as needed)
done
