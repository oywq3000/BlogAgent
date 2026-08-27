#!/usr/bin/env bash
# 博客 AI Agent 独立部署脚本（Windows git-bash / Linux 均可）
# 流程: tar-over-ssh 上传源码（就地覆盖 + code.bak.tar.gz 快照）
#       -> 服务器 docker compose 构建并启动
#       默认只部署源码；传 --sync-config 才同步 Dockerfile/compose/env（MD5 有变化才上传）
# 用法: ./deploy/deploy.sh [--clean] [--rollback] [--sync-config]
set -euo pipefail

# ============ 配置区（按需修改） ============
SERVER_HOST="100.110.148.14"          # 首次部署建议先用服务器 IP
SERVER_USER="oy"                # 非 root 需对 REMOTE_DIR 有写权限
REMOTE_DIR="/home/oy/app/oyblogdeploy/blogagent"
SSH_BIN="C:\Windows\System32\OpenSSH\ssh.exe"       # 中文用户 home 导致 ssh 读密钥失败时，改 /c/Windows/System32/OpenSSH/ssh.exe
SCP_BIN="C:\Windows\System32\OpenSSH\scp.exe"
# ===========================================

SSH_TARGET="${SERVER_USER}@${SERVER_HOST}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_CMD="docker compose"    # compose v1 改为 "docker-compose"

SYNC_CONFIG=0; CLEAN=0; ROLLBACK=0
for arg in "$@"; do
  case "$arg" in
    --sync-config) SYNC_CONFIG=1 ;;
    --clean)       CLEAN=1 ;;
    --rollback)    ROLLBACK=1 ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

# 用 Windows 原生 OpenSSH 时，MSYS 会把 /home/oy 等远端路径误转成 Windows 路径，必须关掉
case "$SSH_BIN" in
  /c/Windows/*) export MSYS_NO_PATHCONV=1 ;;
esac

if [ "$ROLLBACK" = "1" ]; then
  echo "==> 回滚源码（恢复上次构建快照并重建镜像）"
  "${SSH_BIN}" "$SSH_TARGET" "cd $REMOTE_DIR && [ -f code.bak.tar.gz ] || { echo '没有可回滚的快照'; exit 1; } && rm -rf app && tar -xzf code.bak.tar.gz && cd deploy && $COMPOSE_CMD up -d --build"
  exit 0
fi

echo "==> [1/4] 上传源码（tar-over-ssh）"
# 红线: REMOTE_DIR 是 compose build context，就地覆盖；先快照 code.bak.tar.gz 供回滚
"${SSH_BIN}" "$SSH_TARGET" "mkdir -p ${REMOTE_DIR}/deploy"
REMOTE_SCRIPT=$(cat <<EOF
set -e
cd $REMOTE_DIR
[ -f code.bak.tar.gz ] && rm -f code.bak.tar.gz
tar -czf code.bak.tar.gz app requirements.txt .dockerignore 2>/dev/null || true   # 首次部署没有旧源码
if [ "$CLEAN" = "1" ]; then rm -rf app; fi
tar -xzf - -C .
EOF
)
tar -czf - -C "${REPO_ROOT}" app requirements.txt .dockerignore | "${SSH_BIN}" "$SSH_TARGET" "$REMOTE_SCRIPT"

echo "==> [2/4] 同步部署配置"

# --sync-config 时：Dockerfile/compose/env 两端 MD5 比对，有变化才上传
if [ "$SYNC_CONFIG" = "1" ]; then
  CONFIG_FILES=(
    "deploy/Dockerfile|deploy/Dockerfile"
    "deploy/docker-compose.yml|deploy/docker-compose.yml"
    "deploy/.env|deploy/.env"
  )
  for entry in "${CONFIG_FILES[@]}"; do
    IFS='|' read -r local_rel remote_rel <<< "${entry}"
    OLD_MD5=$("${SSH_BIN}" "$SSH_TARGET" "md5sum ${REMOTE_DIR}/${remote_rel} 2>/dev/null | awk '{print \$1}'" || true)
    NEW_MD5=$(md5sum "${REPO_ROOT}/${local_rel}" | awk '{print $1}')
    if [ "$OLD_MD5" != "$NEW_MD5" ]; then
      "${SCP_BIN}" -q "${REPO_ROOT}/${local_rel}" "${SSH_TARGET}:${REMOTE_DIR}/${remote_rel}"
      echo "   ${remote_rel} 已更新"
    else
      echo "   ${remote_rel} 未变化，跳过"
    fi
  done
else
  echo "   跳过（默认只部署源码，compose/env 以服务器现状为准；首次部署请带 --sync-config）"
fi

echo "==> [3/4] 服务器构建镜像并启动（up -d --build 幂等）"
"${SSH_BIN}" "$SSH_TARGET" "cd ${REMOTE_DIR}/deploy && ${COMPOSE_CMD} config -q && ${COMPOSE_CMD} up -d --build"

echo "==> [4/4] 冒烟验证（容器内 /docs -> 200，最多重试 60 秒）"
REMOTE_SMOKE=$(cat <<'EOF'
set -e
for i in $(seq 1 20); do
  code=$(docker exec oy-blog-python-agent python -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8001/docs").status)' 2>/dev/null || true)
  if [ "$code" = "200" ]; then echo "   冒烟通过: /docs -> 200"; exit 0; fi
  sleep 3
done
echo "   冒烟失败: 60 秒后仍未 200，请 docker logs oy-blog-python-agent 排查"
exit 1
EOF
)
"${SSH_BIN}" "$SSH_TARGET" "$REMOTE_SMOKE"

echo ""
echo "==> 部署完成。验证:"
echo "  1. ssh ${SSH_TARGET} 'cd ${REMOTE_DIR}/deploy && docker compose ps'  # 期望 Up"
echo "  2. 内存: ssh ${SSH_TARGET} 'docker stats --no-stream oy-blog-python-agent'  # RSS 约 300-400MB"
echo "  3. 博客前端对话冒烟（Java agent-service 走容器名 oy-blog-python-agent:8001）"
