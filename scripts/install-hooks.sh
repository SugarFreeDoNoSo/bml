#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config --local core.hooksPath .githooks
printf '%s\n' 'Installed local repository hooks.'
