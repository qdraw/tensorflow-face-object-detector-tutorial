#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Did you update the 'tensorflow_models' path in this script?"
read -rsp $'Press any key to continue...\n' -n 1 key
echo ">>>>>"

python3 ~/tensorflow_models/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ssd_mobilenet_v1_face.config \
--trained_checkpoint_prefix model_output/model.ckpt-12262 \
--output_directory model/

# Output:
# Converted 199 variables to const ops.
