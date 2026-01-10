#!/bin/bash
while true; do
  echo "Checking for updates at $(date)"
  
  # Stash any local changes (just in case)
  git stash --quiet
  
  # Pull updates
  git pull origin claude/initial-setup-011CUpNZgwYnxz6wJsGE4hFb
  
  if [ $? -eq 0 ]; then
    echo "✓ Updated successfully"
  else
    echo "✗ Pull failed (you might have conflicts)"
  fi
  
  echo "---"
  sleep 30  # Check every 30 seconds (adjust as needed)
done
