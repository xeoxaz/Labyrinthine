#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
arch_dir="$repo_root/packaging/arch"

if ! command -v makepkg >/dev/null 2>&1; then
  echo "error: makepkg is required to regenerate .SRCINFO" >&2
  exit 1
fi

echo "==> release readiness"
"$repo_root/scripts/check-release-readiness.sh"

echo "==> refresh .SRCINFO"
cd "$arch_dir"
makepkg --printsrcinfo > .SRCINFO

echo "==> package metadata refreshed"
echo "Next steps:"
echo "  1. Review git diff"
echo "  2. Commit and push upstream"
echo "  3. Copy packaging/arch/PKGBUILD and packaging/arch/.SRCINFO into the AUR repo"