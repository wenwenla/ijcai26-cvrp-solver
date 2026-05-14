
This is the official repository for "A Unified Knowledge Embedded Reinforcement Learning-based Framework for Generalized Capacitated Vehicle Routing Problems".

### training

```python3 train_mixed_rf.py --epochs 300 --epoch_size 1280000 --nodes 50 --folder debug-50 --aug 1 --pomo 8  --batch 512 --seed 3333 --div 2 --lr -1```

### evaluate

```python3 evaluate.py```

Currently, you can use python3 evaluate.py to perform performance evaluation. 
If you need to use different datasets or evaluation parameters, please modify **line 24/29** (TODO). 
I apologize for the current disorganized structure of the evaluation code. 
When I have time, I will refactor the related evaluation code using command-line arguments.

By default, the evaluation is conducted on CVRP instances with n=50. If you would like to use other datasets, please download the corresponding data from https://huggingface.co/ai4co/routefinder
 and place the files into the data folder.


### pretrained models

You can download model from https://box.nju.edu.cn/d/8084206b2e5f4afab106/, place ```299.pt``` into ```logs/debug-50``` or ```logs/debug-100```.

### acknowledgement

We thank RouteFinder for the datasets and codes. 
In addition, the prior work https://github.com/wenwenla/ecai-2025-mtsp-solver associated with this project may also be helpful.


Please consider cite our paper.

TODO: bib

