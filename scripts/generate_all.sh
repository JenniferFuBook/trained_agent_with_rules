#!/bin/bash
# Bash script to generate all 6 datasets

# Create output directory
mkdir -p data

# Add node examples
python3 scripts/generate_dataset.py \
  --nodes scripts/nodes.txt \
  --templates scripts/add_templates.txt \
  --output data/add_node.txt \
  --command "ADD_NODE {node}" \
  --count 1000

# Delete node examples
python3 scripts/generate_dataset.py \
  --nodes scripts/nodes.txt \
  --templates scripts/delete_templates.txt \
  --output data/delete_node.txt \
  --command "DELETE_NODE {node}" \
  --count 1000

# Connect nodes examples
python3 scripts/generate_dataset.py \
  --nodes scripts/nodes.txt \
  --templates scripts/connect_templates.txt \
  --output data/connect.txt \
  --command "CONNECT {node_a} {node_b}" \
  --count 1000

# Disconnect nodes examples
python3 scripts/generate_dataset.py \
  --nodes scripts/nodes.txt \
  --templates scripts/disconnect_templates.txt \
  --output data/disconnect.txt \
  --command "DISCONNECT {node_a} {node_b}" \
  --count 1000
