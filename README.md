# inspect-evals GAIA 渐进式评测脚手架

这个仓库用于学习如何用 `inspect-evals` 调用 GAIA 测试集评测大模型。它不是一上来就跑全量数据集，而是按「检查环境 -> 预览命令 -> 跑 1 条 -> 跑小批量 -> 查看日志 -> 再扩大范围」的顺序推进。

参考文档：

- GAIA 任务说明：https://ukgovernmentbeis.github.io/inspect_evals/evals/assistants/gaia/
- Inspect 模型配置：https://inspect.aisi.org.uk/models.html
- Inspect 模型提供方和 OpenAI-compatible 协议：https://inspect.aisi.org.uk/providers.html

## 1. 准备虚拟环境

需要 Python 3.11 或更高版本。

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

GAIA 默认会让模型使用浏览器、bash、python 等工具，并在 Docker sandbox 中执行命令，所以本机还需要安装并启动 Docker。

GAIA 数据集来自 Hugging Face。你需要先在 Hugging Face 申请 GAIA 数据集访问权限，然后准备 token：

```bash
cp config/model_profiles.example.toml config/model_profiles.local.toml
touch .env
```

在 `.env` 中写入你的密钥，例如：

```bash
HF_TOKEN=hf_xxx
OPENAI_API_KEY=sk_xxx
DEEPSEEK_API_KEY=sk_xxx
```

`.env` 和 `config/model_profiles.local.toml` 已被忽略，不会提交到 git。

## 2. 配置模型 Profile

编辑 `config/model_profiles.local.toml`。每个 profile 至少需要一个 `model`：

```toml
[profiles.my_openai]
description = "我的 OpenAI 官方接口"
model = "openai/gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
```

OpenAI-compatible 服务使用 Inspect 的 `openai-api/<provider>/<model>` 命名：

```toml
[profiles.my_provider]
model = "openai-api/my-provider/my-model"
api_key_env = "MY_PROVIDER_API_KEY"
base_url_env = "MY_PROVIDER_BASE_URL"
base_url = "https://api.example.com/v1"

[profiles.my_provider.model_args]
emulate_tools = true
```

如果 provider 名是 `my-provider`，Inspect 默认也会按 `MY_PROVIDER_API_KEY`、`MY_PROVIDER_BASE_URL` 这种约定读取环境变量。脚本会把 profile 里的 `base_url` 写入对应环境变量，便于不同模型服务各用各的 base URL。

如果 GAIA 的真实网页访问需要代理，可以在 profile 下加 `proxy`。Docker Desktop 场景下，sandbox 容器里的 `127.0.0.1` 不是宿主机，所以代理地址通常要写 `host.docker.internal`：

```toml
[profiles.deepseek_v4_flash.proxy]
all = "socks5://host.docker.internal:7890"
no_proxy = "localhost,127.0.0.1"
```

也可以分别配置 HTTP/HTTPS：

```toml
[profiles.deepseek_v4_flash.proxy]
http = "http://host.docker.internal:7890"
https = "http://host.docker.internal:7890"
no_proxy = "localhost,127.0.0.1"
```

脚本会把代理设置为 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 及小写形式，并通过 `inspect eval --env` 传给 GAIA sandbox 内的浏览器、bash、python 工具。Hugging Face 数据集下载和模型 SDK 调用也会继承这些代理。

DeepSeek 官方 OpenAI-compatible API 的 base URL 是 `https://api.deepseek.com`。当前模型名示例：

```toml
[profiles.deepseek_v4_flash]
model = "openai-api/deepseek/deepseek-v4-flash"
api_key_env = "DEEPSEEK_API_KEY"
base_url_env = "DEEPSEEK_BASE_URL"
base_url = "https://api.deepseek.com"

[profiles.deepseek_v4_pro]
model = "openai-api/deepseek/deepseek-v4-pro"
api_key_env = "DEEPSEEK_API_KEY"
base_url_env = "DEEPSEEK_BASE_URL"
base_url = "https://api.deepseek.com"
```

## 3. 先检查，不运行评测

列出可用 profile：

```bash
python scripts/gaia_eval.py profiles
```

检查某个 profile 的前置条件：

```bash
python scripts/gaia_eval.py doctor --profile my_openai
```

这一步会检查 `inspect` 命令、模型 key、`HF_TOKEN` 和 Docker，但不会调用模型。

## 4. 预览将要执行的命令

```bash
python scripts/gaia_eval.py plan --profile my_openai
```

默认计划是：

- GAIA Level 1
- validation split
- 只跑 1 条样本
- 并发数 1
- temperature 0

你可以先确认输出的 `inspect eval ...` 命令是否符合预期。

## 5. 跑第一个最小样本

```bash
python scripts/gaia_eval.py run --profile my_openai
```

等价于跑已安装 `inspect-evals` 包中的 `gaia.py@gaia_level1` 任务的 1 条 validation 样本。日志默认写到 `logs/inspect/`。

查看最近生成的日志文件：

```bash
python scripts/gaia_eval.py logs
```

启动 Web 日志查看器：

```bash
python scripts/gaia_eval.py view
```

## 6. 逐步扩大范围

先跑 3 条 Level 1：

```bash
python scripts/gaia_eval.py run --profile my_openai --level 1 --limit 3
```

再尝试 Level 2：

```bash
python scripts/gaia_eval.py run --profile my_openai --level 2 --limit 3
```

指定 validation split 中的一段样本：

```bash
python scripts/gaia_eval.py run --profile my_openai --level 1 --limit 10-20
```

增加并发前先确认你的模型服务限流：

```bash
python scripts/gaia_eval.py run --profile my_openai --level 1 --limit 10 --max-connections 2
```

## 7. 全量运行

全量运行成本高、耗时长，也更容易触发限流。脚本故意要求你显式确认。注意这里 `--limit ""` 表示不传样本限制：

```bash
python scripts/gaia_eval.py run --profile my_openai --level all --limit "" --confirm-full
```

也可以分层运行：

```bash
python scripts/gaia_eval.py run --profile my_openai --level 1 --limit "" --confirm-full
python scripts/gaia_eval.py run --profile my_openai --level 2 --limit "" --confirm-full
python scripts/gaia_eval.py run --profile my_openai --level 3 --limit "" --confirm-full
```

## 常见问题

如果 `doctor` 提示缺少 `HF_TOKEN`，说明还没有配置 Hugging Face token，或 `.env` 不在项目根目录。

如果模型能聊天但 GAIA 失败，优先检查该模型服务是否支持 tool calling。GAIA 需要模型使用工具；对不支持原生 tool calling 的 OpenAI-compatible 服务，可以尝试在 profile 的 `model_args` 中设置 `emulate_tools = true`。

如果 Docker 相关错误出现，先确认 Docker Desktop 或 Docker Engine 已启动，再重新运行单样本测试。
