# training_checklist

一些和「训练/数据检查」相关的小脚本示例，目前包含：

- `mcp/`：一个最小可运行的 MCP Server demo（基于 `fastmcp`）
- `sft/`：一个用于可视化多轮 SFT 数据 `loss_mask` 的检查脚本（依赖 `verl` 的数据集实现）

## 目录结构

- `mcp/mcp_demo.py`：启动一个包含 `greet` 工具的 MCP Server
- `mcp/mcp_servers.json`：示例 MCP client 配置（指向本机 `8090` 端口）
- `mcp/mcp_tools.yaml`：工具清单占位文件（当前为空）
- `sft/loss_mask_check.py`：对多轮 SFT parquet 数据进行 tokenize，并用颜色展示 `loss_mask`（绿=参与 loss，灰=不参与）

## 环境准备

本项目没有固定依赖清单（未提供 `requirements.txt` / `pyproject.toml`）。你可以按需安装：

- MCP demo：
  - `pip install fastmcp`
- `loss_mask` 检查脚本（按你的环境补齐即可）：
  - `pip install transformers omegaconf rich torch`
  - 以及可用的 `verl`（需要能 import：`verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`）

## MCP Demo（mcp/）

1. 安装依赖：`pip install fastmcp`
2. 启动服务：
   - `python mcp/mcp_demo.py`
3. 默认会在本机启动 MCP 服务；`mcp/mcp_servers.json` 提供了一个指向 `http://127.0.0.1:8090/mcp` 的示例配置（具体端口/路径以你的 MCP client 与 `fastmcp` 运行方式为准）。

## SFT loss_mask 可视化（sft/）

`sft/loss_mask_check.py` 会：

- 加载 `MODEL_PATH` 对应的 tokenizer
- 读取 `DATA_PATH` 指向的多轮对话 parquet
- 将每条样本解码出来，并用颜色标注 `loss_mask`

使用方法：

1. 打开并修改脚本顶部常量：
   - `MODEL_PATH`：本地/远程模型路径（用于 `AutoTokenizer.from_pretrained`）
   - `DATA_PATH`：待检查的 parquet 文件路径
   - `MAX_SAMPLES`：最多检查多少条样本
2. 运行：
   - `python sft/loss_mask_check.py`

说明：

- 该脚本目前使用绝对路径作为默认值（更像是“个人检查脚本模板”）。如果你想在不同机器/目录复用，建议把这些常量改成命令行参数或环境变量。

