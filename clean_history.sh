#!/bin/bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch upload_to_hub.py .env" \
  --prune-empty --tag-name-filter cat -- --all 