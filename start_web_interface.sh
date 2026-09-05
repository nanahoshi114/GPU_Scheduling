#!/usr/bin/env bash
# 快速启动 GPU 拓扑感知调度器 Web 界面。
# 用法：./start_web_interface.sh
# 可选环境变量：HOST（默认 127.0.0.1）、PORT（默认 8000）
# 强制重编 C++ 扩展：./start_web_interface.sh --rebuild

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VENV="${ROOT}/.venv"
REBUILD=0

for arg in "$@"; do
  case "${arg}" in
    --rebuild|-r) REBUILD=1 ;;
    -h|--help)
      echo "用法: $0 [--rebuild]"
      echo "  HOST=0.0.0.0 PORT=8080 $0"
      exit 0
      ;;
    *)
      echo "未知参数: ${arg}" >&2
      echo "用法: $0 [--rebuild]" >&2
      exit 1
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3（需要 3.9+）" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "创建虚拟环境 ${VENV}"
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

need_install=0
if [[ "${REBUILD}" -eq 1 ]]; then
  need_install=1
elif ! python -c "import gpu_scheduler, uvicorn" >/dev/null 2>&1; then
  need_install=1
fi

if [[ "${need_install}" -eq 1 ]]; then
  echo "安装 gpu-scheduler 与 Web 依赖..."
  pip install -e .
fi

echo "Web 界面：http://${HOST}:${PORT}"
echo "交互调度 / 策略对比。Ctrl+C 停止。"
exec uvicorn python.web.app:app --reload --host "${HOST}" --port "${PORT}"
