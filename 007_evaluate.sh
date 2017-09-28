#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Did you update the 'tensorflow_models' path in this script?"
read -rsp $'Press any key to continue...\n' -n 1 key
echo ">>>>>"

python3 ~/tensorflow_models/object_detection/eval.py --logtostderr --pipeline_config_path=ssd_mobilenet_v1_face.config  --checkpoint_dir=model_output --eval_dir=eval

# Output:
# INFO:tensorflow:Restoring parameters from /Users/dionvanvelde/Desktop/models/wider/train/model.ckpt-14378
# INFO:tensorflow:Restoring parameters from /Users/dionvanvelde/Desktop/models/wider/train/model.ckpt-14378
# WARNING:root:The following classes have no ground truth examples: 0
