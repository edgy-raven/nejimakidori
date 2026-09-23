#!/bin/bash
set -euo pipefail

remote="${THOTH_SSH:-fen@34.123.50.212}"
local_port="${LOCAL_PORT:-8793}"
remote_port="${REMOTE_PORT:-8793}"

exec autossh \
    -M 0 \
    -NT \
    -o "ExitOnForwardFailure=yes" \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -R "127.0.0.1:${remote_port}:127.0.0.1:${local_port}" \
    "$remote"
