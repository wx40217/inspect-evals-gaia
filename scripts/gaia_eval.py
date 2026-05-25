#!/usr/bin/env python3
"""循序渐进运行 inspect-evals 的 GAIA 评测。

这个脚本只是 Inspect CLI 的一层学习型包装：
- 统一读取不同模型协议的 profile；
- 在真正执行前可以先 doctor/plan；
- run 的默认值很保守，只跑 GAIA Level 1 的 1 条 validation 样本。
"""

from __future__ import annotations

import argparse
import json
import os
import site
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "model_profiles.local.toml"
EXAMPLE_CONFIG = ROOT / "config" / "model_profiles.example.toml"
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_LOG_DIR = ROOT / "logs" / "inspect"
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

GAIA_TASK_NAMES = {
    "all": "gaia",
    "1": "gaia_level1",
    "2": "gaia_level2",
    "3": "gaia_level3",
}


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    model: str
    api_key_env: str | None
    api_key: str | None
    base_url_env: str | None
    base_url: str | None
    proxy: dict[str, str]
    model_args: dict[str, Any]
    extra_env: dict[str, str]


def mask_secret(value: str | None) -> str:
    """展示配置时隐藏密钥，避免复制终端输出时泄露。"""
    if not value:
        return "<未设置>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_profiles(config_path: Path) -> dict[str, Profile]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}\n"
            f"先执行：cp {EXAMPLE_CONFIG.relative_to(ROOT)} {DEFAULT_CONFIG.relative_to(ROOT)}"
        )

    raw_profiles = read_toml(config_path).get("profiles", {})
    profiles: dict[str, Profile] = {}
    for name, raw in raw_profiles.items():
        if "model" not in raw:
            raise ValueError(f"profile {name!r} 缺少 model 字段")

        extra_env = raw.get("env", {})
        if not isinstance(extra_env, dict):
            raise ValueError(f"profile {name!r} 的 env 必须是 TOML table")
        proxy = raw.get("proxy", {})
        if not isinstance(proxy, dict):
            raise ValueError(f"profile {name!r} 的 proxy 必须是 TOML table")

        profiles[name] = Profile(
            name=name,
            description=str(raw.get("description", "")),
            model=str(raw["model"]),
            api_key_env=raw.get("api_key_env"),
            api_key=raw.get("api_key"),
            base_url_env=raw.get("base_url_env"),
            base_url=raw.get("base_url"),
            proxy={str(k): str(v) for k, v in proxy.items()},
            model_args=dict(raw.get("model_args", {})),
            extra_env={str(k): str(v) for k, v in extra_env.items()},
        )
    return profiles


def load_env_file(path: Path, env: dict[str, str]) -> None:
    """读取简单 .env 文件；只支持 KEY=value，足够放 API key 和 HF_TOKEN。"""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in env:
            env[key] = value


def build_env(profile: Profile, env_file: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if env_file is not None:
        load_env_file(env_file, env)

    # profile 中的显式值用于本地实验；真实项目更推荐只写 env var 名，值放 .env。
    if profile.api_key_env and profile.api_key:
        env[profile.api_key_env] = profile.api_key
    if profile.base_url_env and profile.base_url:
        env[profile.base_url_env] = profile.base_url
    env.update(proxy_env(profile, env))
    env.update(profile.extra_env)
    return env


def proxy_env(profile: Profile, env: dict[str, str], include_ambient: bool = True) -> dict[str, str]:
    """整理代理环境变量，同时设置大小写形式以兼容不同工具。"""
    proxy = profile.proxy
    result: dict[str, str] = {}

    http_proxy = proxy.get("http") or proxy.get("http_proxy") or proxy.get("HTTP_PROXY")
    https_proxy = proxy.get("https") or proxy.get("https_proxy") or proxy.get("HTTPS_PROXY")
    all_proxy = proxy.get("all") or proxy.get("all_proxy") or proxy.get("ALL_PROXY")
    no_proxy = proxy.get("no_proxy") or proxy.get("NO_PROXY")

    # 支持只配置 all = "socks5://127.0.0.1:7890" 的简写。
    if all_proxy and not http_proxy:
        http_proxy = all_proxy
    if all_proxy and not https_proxy:
        https_proxy = all_proxy

    values = {
        "HTTP_PROXY": http_proxy,
        "HTTPS_PROXY": https_proxy,
        "ALL_PROXY": all_proxy,
        "NO_PROXY": no_proxy,
    }
    for key, value in values.items():
        if not value:
            continue
        result[key] = value
        result[key.lower()] = value

    if include_ambient:
        # 如果 profile 没配置代理，但 shell/.env 已经有代理，也让宿主进程继承。
        for key in PROXY_ENV_NAMES:
            if key in env and env[key] and key not in result:
                result[key] = env[key]
    return result


def proxy_env_for_cli(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in PROXY_ENV_NAMES if env.get(key)}


def sandbox_proxy_env(profile: Profile) -> dict[str, str]:
    """只把 profile 中显式配置的代理传进 sandbox，避免误传宿主机 127.0.0.1。"""
    return proxy_env(profile, {}, include_ambient=False)


def looks_like_loopback_proxy(value: str) -> bool:
    return "://127.0.0.1" in value or "://localhost" in value


def inspect_executable() -> str:
    """优先使用当前虚拟环境里的 inspect，找不到时再依赖 PATH。"""
    venv_inspect = ROOT / ".venv" / "bin" / "inspect"
    if venv_inspect.exists():
        return str(venv_inspect)
    return shutil.which("inspect") or "inspect"


def gaia_task_file() -> Path:
    """定位虚拟环境或当前 Python 环境中安装的 inspect_evals GAIA 任务文件。"""
    candidates = [
        ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        *[Path(path) for path in site.getsitepackages()],
        Path(site.getusersitepackages()),
    ]
    for base in candidates:
        task_file = base / "inspect_evals" / "gaia" / "gaia.py"
        if task_file.exists():
            return task_file
    raise FileNotFoundError(
        "找不到 inspect_evals/gaia/gaia.py。请先运行 bash scripts/setup_venv.sh 安装依赖。"
    )


def gaia_task_ref(level: str) -> str:
    return f"{gaia_task_file()}@{GAIA_TASK_NAMES[level]}"


def value_for_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def task_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for key in ("split", "max_attempts"):
        value = getattr(args, key, None)
        if value is not None:
            result += ["-T", f"{key}={value}"]
    if args.instance_id:
        result += ["-T", f"instance_ids={args.instance_id}"]
    return result


def build_eval_command(profile: Profile, args: argparse.Namespace, env: dict[str, str] | None = None) -> list[str]:
    task = gaia_task_ref(args.level)
    cmd = [
        inspect_executable(),
        "eval",
        task,
        "--model",
        profile.model,
        "--log-dir",
        str(args.log_dir),
        "--max-connections",
        str(args.max_connections),
        "--temperature",
        str(args.temperature),
    ]

    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.sample_id:
        cmd += ["--sample-id", str(args.sample_id)]
    if args.message_limit:
        cmd += ["--message-limit", str(args.message_limit)]
    if profile.base_url and not profile.base_url_env:
        # 只有没指定 provider 专用 base_url_env 时，才走通用 CLI 参数。
        cmd += ["--model-base-url", profile.base_url]

    for key, value in profile.model_args.items():
        cmd += ["-M", f"{key}={value_for_cli(value)}"]

    sandbox_proxies = sandbox_proxy_env(profile)
    if env and sandbox_proxies:
        for key, value in sandbox_proxies.items():
            cmd += ["--env", f"{key}={value}"]

    cmd += task_args(args)
    return cmd


def print_command(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in cmd))


def run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    """运行 Inspect 子命令；用户按 Ctrl+C 时安静退出。"""
    try:
        return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


def require_profile(args: argparse.Namespace) -> tuple[Profile, dict[str, Profile]]:
    profiles = load_profiles(args.config)
    if args.profile not in profiles:
        names = ", ".join(sorted(profiles)) or "<空>"
        raise SystemExit(f"找不到 profile：{args.profile}\n可用 profile：{names}")
    return profiles[args.profile], profiles


def cmd_profiles(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.config)
    for profile in profiles.values():
        print(f"{profile.name}: {profile.model}")
        if profile.description:
            print(f"  说明：{profile.description}")
        if profile.api_key_env:
            print(f"  Key 环境变量：{profile.api_key_env}")
        if profile.base_url_env:
            print(f"  Base URL 环境变量：{profile.base_url_env}")
        elif profile.base_url:
            print(f"  Base URL：{profile.base_url}")
        if profile.proxy:
            print("  代理：已配置")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    profile, _ = require_profile(args)
    env = build_env(profile, args.env_file)
    errors: list[str] = []

    print(f"配置文件：{args.config}")
    print(f"Profile：{profile.name}")
    print(f"模型：{profile.model}")

    inspect_bin = inspect_executable()
    print(f"Inspect CLI：{inspect_bin}")
    if shutil.which(inspect_bin) is None and not Path(inspect_bin).exists():
        errors.append("找不到 inspect 命令，请先运行 bash scripts/setup_venv.sh 并激活虚拟环境")

    if profile.api_key_env:
        print(f"{profile.api_key_env}：{mask_secret(env.get(profile.api_key_env))}")
        if not env.get(profile.api_key_env):
            errors.append(f"缺少 {profile.api_key_env}；可写入 .env 或先 export")

    if profile.base_url_env:
        print(f"{profile.base_url_env}：{env.get(profile.base_url_env) or '<未设置>'}")
    elif profile.base_url:
        print(f"通用 --model-base-url：{profile.base_url}")

    host_proxies = proxy_env_for_cli(env)
    sandbox_proxies = sandbox_proxy_env(profile)
    if host_proxies:
        print("宿主进程代理环境变量：")
        for key, value in host_proxies.items():
            print(f"  {key}={value}")
    else:
        print("宿主进程代理环境变量：<未设置>")
    if sandbox_proxies:
        print("Sandbox 代理环境变量：")
        for key, value in sandbox_proxies.items():
            print(f"  {key}={value}")
            if looks_like_loopback_proxy(value):
                errors.append(
                    f"{key} 使用了 localhost/127.0.0.1；Docker sandbox 通常应改为 host.docker.internal"
                )
    else:
        print("Sandbox 代理环境变量：<未设置>")

    print(f"HF_TOKEN：{mask_secret(env.get('HF_TOKEN'))}")
    if not env.get("HF_TOKEN"):
        errors.append("GAIA 数据集需要 Hugging Face 授权，缺少 HF_TOKEN")

    docker = shutil.which("docker")
    print(f"Docker：{docker or '<未找到>'}")
    if not docker:
        errors.append("GAIA 默认 sandbox 需要 Docker")

    if errors:
        print("\n需要处理：")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\n基础检查通过。下一步可以运行 plan 或 run。")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    profile, _ = require_profile(args)
    env = build_env(profile, args.env_file)
    cmd = build_eval_command(profile, args, env)

    print("将使用的环境变量：")
    if profile.api_key_env:
        print(f"  {profile.api_key_env}={mask_secret(env.get(profile.api_key_env))}")
    if profile.base_url_env:
        print(f"  {profile.base_url_env}={env.get(profile.base_url_env) or '<未设置>'}")
    if env.get("HF_TOKEN"):
        print(f"  HF_TOKEN={mask_secret(env.get('HF_TOKEN'))}")
    for key, value in proxy_env_for_cli(env).items():
        print(f"  {key}={value}")
    sandbox_proxies = sandbox_proxy_env(profile)
    if sandbox_proxies:
        print("将传入 sandbox 的代理：")
        for key, value in sandbox_proxies.items():
            print(f"  {key}={value}")
    else:
        print("将传入 sandbox 的代理：<未设置；如需让浏览器走代理，请在 profile.proxy 中配置>")

    print("\n将执行的命令：")
    print_command(cmd)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    profile, _ = require_profile(args)
    full_run_requested = args.limit is None or args.limit == ""
    if full_run_requested and not args.confirm_full:
        raise SystemExit(
            "你正在请求没有 --limit 的运行。"
            "如果确定要跑全量，请加 --confirm-full。"
        )

    env = build_env(profile, args.env_file)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_eval_command(profile, args, env)

    print("开始执行：")
    print_command(cmd)
    print()
    return run_subprocess(cmd, env=env)


def cmd_logs(args: argparse.Namespace) -> int:
    log_dir = args.log_dir
    if not log_dir.exists():
        print(f"日志目录不存在：{log_dir}")
        return 1

    logs = sorted(log_dir.rglob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print(f"没有找到 .eval 日志：{log_dir}")
        return 1

    for path in logs[: args.limit]:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(rel)
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    cmd = [inspect_executable(), "view", "--log-dir", str(args.log_dir)]
    if args.host:
        cmd += ["--host", args.host]
    if args.port:
        cmd += ["--port", str(args.port)]

    print("启动 Inspect 日志查看器：")
    print_command(cmd)
    print()
    return run_subprocess(cmd)


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else EXAMPLE_CONFIG,
        help="模型 profile 配置文件",
    )


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="要使用的模型 profile 名称")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.exists() else None,
        help="可选 .env 文件；默认自动读取项目根目录 .env",
    )


def add_run_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level", choices=sorted(GAIA_TASK_NAMES), default="1", help="GAIA 难度层级")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="数据集 split")
    parser.add_argument("--limit", default="1", help="样本数量或范围；默认只跑 1 条")
    parser.add_argument("--sample-id", help="Inspect 的样本 id 过滤")
    parser.add_argument("--instance-id", help="GAIA 原始 instance_id 过滤")
    parser.add_argument("--max-attempts", type=int, default=1, help="默认 solver 的最大提交次数")
    parser.add_argument("--message-limit", type=int, help="单个样本的消息轮数上限")
    parser.add_argument("--max-connections", type=int, default=1, help="并发连接数；学习阶段建议保持 1")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Inspect 日志目录")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="渐进式运行 inspect-evals GAIA 评测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles = subparsers.add_parser("profiles", help="列出配置文件中的模型 profile")
    add_common_config_args(profiles)
    profiles.set_defaults(func=cmd_profiles)

    doctor = subparsers.add_parser("doctor", help="检查虚拟环境、key、HF_TOKEN、Docker 等前置条件")
    add_common_config_args(doctor)
    add_profile_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    plan = subparsers.add_parser("plan", help="只打印将要执行的 inspect eval 命令")
    add_common_config_args(plan)
    add_profile_args(plan)
    add_run_shape_args(plan)
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="执行 GAIA 评测；默认只跑 Level 1 的 1 条样本")
    add_common_config_args(run)
    add_profile_args(run)
    add_run_shape_args(run)
    run.add_argument("--confirm-full", action="store_true", help="允许无 limit 跑 inspect_evals/gaia 全量任务")
    run.set_defaults(func=cmd_run)

    logs = subparsers.add_parser("logs", help="列出最近的 Inspect .eval 日志")
    logs.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    logs.add_argument("--limit", type=int, default=10)
    logs.set_defaults(func=cmd_logs)

    view = subparsers.add_parser("view", help="启动 inspect view 查看日志")
    view.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    view.add_argument("--host")
    view.add_argument("--port", type=int)
    view.set_defaults(func=cmd_view)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
