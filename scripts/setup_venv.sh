#!/usr/bin/env bash
set -euo pipefail

# 创建本项目使用的虚拟环境，并安装 GAIA 评测所需依赖。
# 用法：
#   bash scripts/setup_venv.sh

cd "$(dirname "$0")/.."

python3 - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("需要 Python 3.11 或更高版本，因为脚本使用标准库 tomllib 读取 TOML 配置。")
PY

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "虚拟环境已准备好："
echo "  source .venv/bin/activate"
echo "  python scripts/gaia_eval.py doctor --profile openai"
