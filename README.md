# dsv4flash-inference

这个仓库整理了 DeepSeek-V4-Flash 四卡推理优化的源码补丁、固定输入和运行脚本，用于复现约 **3250 输出 tokens/s** 的实验。

## 实验内容

原实验使用 4 张 NVIDIA RTX PRO 6000 Blackwell（SM120）。vLLM 使用 **V1 引擎和 V2 GPU 模型执行器**；TP1 / DP4 表示每张卡运行一个完整模型副本，共 4 个副本。

R6 是本实验对双微批执行方案的命名：在各副本协调后，当调度步至少包含 4096 个 token 时允许拆成两个微批。补丁还包含 SM120 原生索引内核和共享前缀 KV 缓存恢复点的保护逻辑，需要配合仓库中的完整配置使用。

| 配置 | 有效输出吞吐（tokens/s） | 成功请求 |
| --- | ---: | ---: |
| 同批实验基线 | 2864.01 | 256/256 |
| R6 第一次启动 | 3262.11 | 256/256 |
| R6 第二次启动 | 3237.10 | 256/256 |

两次 R6 的中位数为 **3249.61 tokens/s**，比上述基线高 **13.46%**。这里的有效输出吞吐 = 成功请求的输出 token 总数 ÷ 整轮正式测速的墙钟时间。只有两次 R6 启动，结果仅代表这组环境和负载。

正式测速有 128 个独立会话，每个请求输入 131072 个 token、固定输出 1024 个 token；同一会话串行发送两轮，会话之间并发运行，不等待所有会话完成第一轮才发第二轮。原实验每个正式请求命中 126976 个缓存 token。使用 Completion 接口和 token ID 输入，`temperature=0`、`seed=20261104`、`ignore_eos=true`。

基线和 R6 的单卡显存采样最大值分别约为 91.7 和 94.9 GiB；这是定期采样值。两次 R6 正式测速分别出现 26 和 19 条可恢复的显存分配重试警告，请求均完成。

## 环境要求

- Linux x86_64、Python 3.12、`uv`、GNU `patch`，以及可用的 CUDA 驱动；原实验为上述四卡配置。
- 自行准备 DeepSeek-V4-Flash-0731 权重，模型版本（revision）为 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`。
- 服务端：vLLM 0.32.0、PyTorch 2.13.0、FlashInfer 0.7.0.post1、b12x 1.5.1。
- 测速客户端：AIPerf 0.13.0，使用独立 Python 环境。

补丁针对 PyPI 的 x86_64 vLLM 0.32.0 安装包。仓库不包含模型权重；输入文件包含原实验的 384 条固定 token ID 请求（128 条预热、256 条正式请求），来自公开的 SWE-bench 源轨迹，不包含轨迹原文或模型输出。

## 复现步骤

### 1. 安装依赖并准备补丁

以下命令均在仓库根目录执行。服务端安装包需要与本机 CUDA 环境兼容。

```bash
git clone https://github.com/laonongmin666/dsv4flash-inference.git
cd dsv4flash-inference

uv venv --python 3.12 .venv-server
uv pip install --python .venv-server/bin/python -r requirements-server.txt
uv venv --python 3.12 .venv-client
uv pip install --python .venv-client/bin/python -r requirements-client.txt
python3 reproduce.py prepare
```

`prepare` 会把 vLLM 和 AIPerf 复制到 `.runtime/` 后应用补丁，不修改环境中安装的原包。

### 2. 启动服务

替换下面的模型路径和 GPU 编号，只使用分配给自己的卡。第一条命令仅打印启动命令和配置；第二条加上 `--execute` 后才真正启动。

```bash
python3 reproduce.py serve \
  --model /path/to/DeepSeek-V4-Flash-0731 \
  --gpus 0,1,2,3 \
  --arm r6 \
  --output runs/r6-01

python3 reproduce.py serve \
  --model /path/to/DeepSeek-V4-Flash-0731 \
  --gpus 0,1,2,3 \
  --arm r6 \
  --output runs/r6-01 \
  --execute
```

服务日志写入 `runs/r6-01/serve.log`，启动终端需要保持运行。

### 3. 运行测速

等待模型加载完成后，在同一台机器的第二个终端进入仓库，先检查服务，再开始测速：

```bash
curl --fail http://127.0.0.1:8076/health
python3 reproduce.py bench --output runs/r6-01
```

`/health` 检查失败时先查看服务日志，等服务就绪后再运行 `bench`。测速会先让 4 个副本并行预热，各副本内部串行发送 32 条请求，再运行 128 个并发会话的正式测速。

结果位于 `runs/r6-01/measurements/performance-summary.json`：`metrics.effective_output_tok_s` 是有效输出吞吐，`coverage` 中记录成功率和失败原因。

**每轮测速都要重新启动服务并使用新的输出目录**，例如 `runs/r6-02`，避免已有缓存影响结果。结束本轮后，在服务终端按 Ctrl+C 停止。运行基线时，把启动命令中的 `--arm r6` 改为 `--arm baseline`，并更换输出目录；其余配置保持一致。

启动器会拒绝使用已占用的 GPU，但仍需自行确认卡的使用权限。其他硬件、软件版本或负载下的速度需要重新测量。

## 文件说明

- `reproduce.py`：统一入口，提供 `prepare`、`serve`、`bench` 三个命令。
- `patches/`：vLLM 与 AIPerf 的源码补丁。
- `data/`：固定输入和请求配置。
- `client/`：测速客户端及指标计算代码。
- `requirements-*.txt`：服务端和客户端依赖。

仓库自有代码采用 Apache-2.0 许可证。vLLM 和 AIPerf 保留各自许可证；AIPerf 的第三方说明见 `patches/aiperf-LICENSE` 和 `patches/aiperf-ATTRIBUTIONS.md`。
