TASK=$1
AGENT=$2
ASSETS_ROOT=./ravens_torch/environments/assets/

python ravens_torch/demos.py  --assets_root=$ASSETS_ROOT --task=${TASK} --mode=train --n=1000
python ravens_torch/demos.py  --assets_root=$ASSETS_ROOT --task=${TASK} --mode=test  --n=100

python ravens_torch/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1
python ravens_torch/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10
python ravens_torch/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100
python ravens_torch/train.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000

python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=1000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=2000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=5000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=10000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=20000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=40000

python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=1000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=2000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=5000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=10000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=20000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=40000

python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=1000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=2000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=5000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=10000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=20000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=40000

python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=1000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=2000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=5000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=10000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=20000
python ravens_torch/test.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=40000

python ravens_torch/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1
python ravens_torch/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=10
python ravens_torch/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=100
python ravens_torch/plot.py  --assets_root=$ASSETS_ROOT --task=${TASK} --agent=${AGENT} --n_demos=1000
