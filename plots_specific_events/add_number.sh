#!/bin/bash

# Loop through all .png files in the current directory
for file in *.png; do
  # Check if the file exists to avoid errors when no .png files are present
  if [ -f "$file" ]; then
    mv "$file" "11_$file"
  fi
done
