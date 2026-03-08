#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> cargo test"
cargo test --manifest-path "$repo_root/Cargo.toml"

echo "==> cargo build --release"
cargo build --manifest-path "$repo_root/Cargo.toml" --release

echo "==> cargo run -- --help"
cargo run --manifest-path "$repo_root/Cargo.toml" -- --help >/dev/null

echo "==> cargo run -- --version"
cargo run --manifest-path "$repo_root/Cargo.toml" -- --version >/dev/null

echo "==> cargo run -- generate --width 8 --height 4 --seed 42"
cargo run --manifest-path "$repo_root/Cargo.toml" -- generate --width 8 --height 4 --seed 42 >/dev/null

echo "==> PKGBUILD syntax"
bash -n "$repo_root/packaging/arch/PKGBUILD"

echo "==> PKGBUILD.stable syntax"
bash -n "$repo_root/packaging/arch/PKGBUILD.stable"

if command -v makepkg >/dev/null 2>&1; then
  echo "==> .SRCINFO consistency"
  diff -u "$repo_root/packaging/arch/.SRCINFO" <(
    cd "$repo_root/packaging/arch"
    makepkg --printsrcinfo
  )
else
  echo "==> skipping .SRCINFO consistency check (makepkg not installed)"
fi

echo "Release readiness checks passed."