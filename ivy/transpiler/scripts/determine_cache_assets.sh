#!/bin/bash

# Install dependencies
python3 -m pip install cryptography transformers black

# Install the local ivy_repo
cd /ivy/tracer-transpiler/ivy_repo
python3 -m pip install --user -e .

cd ..

# Generate matrices using Python script
cp source_to_source_translator/caching/generate_matrices.py .
torch_frontend_functions_matrix=$(python3 generate_matrices.py torch_frontend_functions_matrix | tr -d '\n')
torch_layers_matrix=$(python3 generate_matrices.py torch_layers_matrix | tr -d '\n')
ivy_functions_matrix=$(python3 generate_matrices.py ivy_functions_matrix | tr -d '\n')

# Calculate batch counts using Python script
cp source_to_source_translator/caching/calculate_batch_counts.py .
torch_frontend_functions_batch_count=$(python3 calculate_batch_counts.py "$torch_frontend_functions_matrix" "$TORCH_FRONTEND_FUNCS_BATCH_SIZE")
torch_layers_batch_count=$(python3 calculate_batch_counts.py "$torch_layers_matrix" "$TORCH_LAYERS_BATCH_SIZE")
ivy_functions_batch_count=$(python3 calculate_batch_counts.py "$ivy_functions_matrix" "$IVY_FUNCS_BATCH_SIZE")

# Write outputs to GITHUB_OUTPUT
if [ -n "$GITHUB_OUTPUT" ]; then
  echo -n "torch_frontend_functions_matrix=" >> "$GITHUB_OUTPUT"
  echo "$torch_frontend_functions_matrix" >> "$GITHUB_OUTPUT"
  echo -n "torch_layers_matrix=" >> "$GITHUB_OUTPUT"
  echo "$torch_layers_matrix" >> "$GITHUB_OUTPUT"
  echo -n "ivy_functions_matrix=" >> "$GITHUB_OUTPUT"
  echo "$ivy_functions_matrix" >> "$GITHUB_OUTPUT"
  echo "torch_frontend_functions_batch_count=$torch_frontend_functions_batch_count" >> "$GITHUB_OUTPUT"
  echo "torch_layers_batch_count=$torch_layers_batch_count" >> "$GITHUB_OUTPUT"
  echo "ivy_functions_batch_count=$ivy_functions_batch_count" >> "$GITHUB_OUTPUT"
fi
