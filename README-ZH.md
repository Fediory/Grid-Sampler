[English](README.md) | **中文**

<p align="center">
  <img src="assets/logo.png" alt="Grid Sampler logo" width="140" />
</p>

# Grid Sampler [ICML 2026]

#### See What Matters：面向可泛化视觉-语言-动作模型的可微网格采样剪枝

<p align="center">
  <a href="https://scholar.google.com/citations?user=WljJ2HUAAAAJ">Yixu Feng</a><sup>1</sup>，
  <a href="https://openreview.net/profile?id=~Zinan_Zhao1">Zinan Zhao</a><sup>2</sup>，
  <a href="https://scholar.google.com/citations?user=mBHSbeIAAAAJ">Yanxiang Ma</a><sup>1</sup>，
  <a href="https://openreview.net/profile?id=~Chenghao_Xia1">Chenghao Xia</a><sup>3</sup>，
  <a href="https://scholar.google.com/citations?user=guY3iCsAAAAJ">Chengbin Du</a><sup>3</sup>，
  <a href="https://scholar.google.com/citations?user=m4wbcOsAAAAJ">Yunke Wang</a><sup>1</sup>，
  <a href="https://scholar.google.com/citations?user=N4F_3eoAAAAJ">Chang Xu</a><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup> 悉尼大学（University of Sydney） &nbsp;·&nbsp;
  <sup>2</sup> 香港城市大学（City University of Hong Kong） &nbsp;·&nbsp;
  <sup>3</sup> StellarEdge Robotics
</p>

## 演示 Demo 🎞

> **为什么在 README 里看不到内嵌播放器？** GitHub 会过滤 HTML `<video>`，因此无法在项目主页直接嵌入播放。请点击下表中的链接，在 GitHub 文件页使用**内置 MP4 播放器**观看（需已将 `demo/` 下视频推送到默认分支，如 `main`）。

**域内（In-domain）**

| Pen | Pick | Stack |
|:---:|:---:|:---:|
| [▶ 观看 `GridS_demo_pen.mp4`](demo/GridS_demo_pen.mp4) | [▶ 观看 `GridS_demo_pick.mp4`](demo/GridS_demo_pick.mp4) | [▶ 观看 `GridS_demo_stack.mp4`](demo/GridS_demo_stack.mp4) |

**分布外（OOD）**

| Pen (OOD) | Pick (OOD) | Stack (OOD) |
|:---:|:---:|:---:|
| [▶ 观看 `GridS_demo_pen_ood.mp4`](demo/GridS_demo_pen_ood.mp4) | [▶ 观看 `GridS_demo_pick_ood.mp4`](demo/GridS_demo_pick_ood.mp4) | [▶ 观看 `GridS_demo_stack_ood.mp4`](demo/GridS_demo_stack_ood.mp4) |

## 动态 News 🆕

- **2026.05.10** 基于 openpi 的代码已公开。💎
- **2026.05.01** 论文 *See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model* 已被 **ICML 2026** 接收，欢迎基于 Grid Sampler 的后续工作。🔥

## 路线图 To-Do List ✅

- ✅ 发布集成 Grid Sampler 的 openpi 代码。
- ⬜ 发布集成 Grid Sampler 的 X-VLA（Robotwin）代码。
- ⬜ 发布集成 Grid Sampler 的 LeRobot 真实数据代码。

## 结果 Results 📊

<details>
<summary><b>LIBERO：</b></summary>

![Result 1](assets/main_result1.png)

</details>

<details>
<summary><b>真实环境（Real-world）：</b></summary>

![Result 2](assets/main_result2.png)

</details>

## 1. 核心思想 Core idea 🌑

![Main idea](assets/core_idea.png)

*GridS 视觉 Token 剪枝框架概览。*

**(a) 标准稠密表示：** 输入图像（*H*<sub>R</sub>、*W*<sub>R</sub> 为原始分辨率）经视觉编码器与 ViT 式嵌入（Dosovitskiy et al., 2020）得到稠密视觉 token（*H* × *W* × *C*），保留完整空间细节。

**(b) GridS Token 剪枝模块：** 在显著区域采样稀疏 token（*K* × *C*），包含两个阶段：(1) 全局坐标预测；(2) 带几何注入的网格采样（Grid Sampling with Geometry Injection）。在 *K* ≪ *H* × *W* 时，为下游 Transformer 提供高效表示。

## 2. 测试 Testing 🌒

### openpi

可运行流程均在 **openpi** 仓库中说明。请先在 [`openpi/README.md`](openpi/README.md) 完成 **Installation** 环境配置，并以该文档为主索引：

- **快速策略自检（无真机）：** 在 **Running Inference for a Pre-Trained Model** 一节中的 **「Test inference without a robot」** 段落，指向 [`openpi/examples/simple_client/README.md`](openpi/examples/simple_client/README.md)。
- **Notebook / 通用推理：** 同上章节及其中提到的 `examples/inference.ipynb`。
- **LIBERO 仿真评测：** [`openpi/examples/libero/README.md`](openpi/examples/libero/README.md)（主 README 微调模型表中亦有 LIBERO 相关说明）。
- **ALOHA：** 仿真见 [`openpi/examples/aloha_sim/README.md`](openpi/examples/aloha_sim/README.md)；真机见 [`openpi/examples/aloha_real/README.md`](openpi/examples/aloha_real/README.md)；汇总亦见 [`openpi/README.md`](openpi/README.md) 的 **More Examples**。

## 3. 微调 Finetuning 🌓

### openpi

微调流程遵循 **openpi** 官方路径，完整说明见 [`openpi/README.md`](openpi/README.md) 中的 **[Fine-Tuning Base Models on Your Own Data](openpi/README.md#fine-tuning-base-models-on-your-own-data)**。简要步骤：

1. **数据：** 转为 LeRobot 数据集（LIBERO 示例脚本：[`openpi/examples/libero/convert_libero_data_to_lerobot.py`](openpi/examples/libero/convert_libero_data_to_lerobot.py)）；若仅使用上游配置中已绑定的 LIBERO 数据，通常可跳过自行转换。
2. **配置：** 数据变换与 `TrainConfig` 在 [`openpi/src/openpi/training/config.py`](openpi/src/openpi/training/config.py)（LIBERO 策略见 [`openpi/src/openpi/policies/libero_policy.py`](openpi/src/openpi/policies/libero_policy.py)）。使用 **Grid Sampler** 时请选择模型侧 `grid=True` 的训练配置（如 `pi0_libero_grid*`、`pi05_libero_grid*` 等）。
3. **训练（JAX）：** 先按文档计算归一化统计量，例如 `uv run scripts/compute_norm_stats.py --config-name <your_config>`，再 `uv run scripts/train.py <your_config> --exp-name=...`（GPU 显存相关见同文档中的 `XLA_PYTHON_CLIENT_MEM_FRACTION` 等说明）。
4. **PyTorch 路线：** 若使用 PyTorch 实现，见同 README 的 **[PyTorch Support](openpi/README.md#pytorch-support)**（环境、`train_pytorch.py`、JAX→PyTorch 权重转换等）。

部署微调后的 checkpoint：见上述章节中的 **Spinning up a policy server and running inference**，以及 [`openpi/scripts/serve_policy.py`](openpi/scripts/serve_policy.py) 与 LIBERO 客户端文档 [`openpi/examples/libero/README.md`](openpi/examples/libero/README.md)。

## 4. 联系方式 Contacts 🌔

如有问题，欢迎发邮件或在仓库中提 Issue，感谢反馈与贡献。

**Yixu Feng：** [yfen0429@sydney.edu.au](yfen0429@sydney.edu.au) 或 [fedioryf@gmail.com](fedioryf@gmail.com)

**Zinan Zhao：** [zhao48zinan@gmail.com](zhao48zinan@gmail.com)

也可扫描下方微信二维码联系：

<img src="assets/wechat.jpg" alt="微信二维码" width="220">

## 5. 引用 Citation 🌕

若本工作对您的研究有帮助，欢迎引用：

```
@inproceedings{feng2026gridsampler,
  title     = {See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model},
  author    = {Feng, Yixu and Zhao, Zinan and Ma, Yanxiang and Xia, Chenghao and Du, Chengbin and Wang, Yunke and Xu, Chang},
  booktitle = {Forty-Third International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
