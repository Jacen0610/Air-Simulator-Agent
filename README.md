# Air-Simulator-Agent

- 注意：本项目必须配合Air-Simulator-Go项目使用。
- 这是使用Pytorch实现简单A2C、DQN、PPO理念的RL-Agent项目。
- simulator.proto定义了grpc的调用接口，包含观测状态（Observation）、动作（Action）、每个周期执行函数（Step）和重置函数（Reset）等。如果修改此文件则Go项目那边也许修改并生成对应的依赖文件。

---


## 目录

- [使用](#使用)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 使用
请根据requirements.txt安装相关依赖，推荐使用conda虚拟环境
```bash
#安装依赖
pip install -r requirments.txt
#运行程序
python train.py
```

---
## 贡献

- 项目负责人：J.H.WANG (220246855@seu.edu.cn)
- 指导老师：W.D.MA (mawd@seu.edu.cn)
- 其他贡献者：
  - S.Q.Xu

---
## 许可证

本项目采用 [MIT 许可证](LICENSE) 授权。