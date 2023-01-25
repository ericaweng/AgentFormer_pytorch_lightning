# Pytorch Lightning implementation of [Agentformer](https://github.com/Khrylx/AgentFormer)

install Agentformer dependencies: `pip install -r requirements` (may need a few other requirements; try running, and install accordingly)

to run: `python pl_train.py`

see `pl_train.py` for all options.

pytorch lightning Trainer file: `trainer.py`

utils for visualizing predictions: `viz_utils.py` (use option `--save_viz` when running main file)

original paper:
AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting  
Ye Yuan, Xinshuo Weng, Yanglan Ou, Kris Kitani  
**ICCV 2021**  
[[website](https://www.ye-yuan.com/agentformer)] [[paper](https://arxiv.org/abs/2103.14023)]

cite:
```
@inproceedings{yuan2021agent,
  title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
