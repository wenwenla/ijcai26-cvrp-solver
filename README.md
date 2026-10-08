
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

You can download model from [this link](https://drive.google.com/drive/folders/1xGeB8OxcNSqVg31QUDRdGo5D-1fNymcg?usp=sharing), place ```299.pt``` into ```logs/debug-50``` or ```logs/debug-100```.

### acknowledgement

We thank RouteFinder for the datasets and codes. 

Please consider citing our paper.

```
@inproceedings{ijcai2026p691,
  title     = {A Unified Knowledge Embedded Reinforcement Learning-based Framework for Generalized Capacitated Vehicle Routing Problems},
  author    = {Wang, Wen and Wu, Xiangchen and Wang, Liang and Hu, Hao and Tao, Xianping},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {6208--6216},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/691},
  url       = {https://doi.org/10.24963/ijcai.2026/691},
}
```

