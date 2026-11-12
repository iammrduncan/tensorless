#!/bin/sh
# Install the tensorless launcher from its GitHub release, checked by sha256.
#
#   curl -fsSL https://tensorless.dev/install.sh | sh
#
# Settings (environment variables):
#   TENSORLESS_VERSION      the release to install (default: 0.2.0)
#   TENSORLESS_INSTALL_DIR  where to put the binary (default: $HOME/.local/bin)
#
# It installs one file, the tensorless binary, and changes nothing else. Linux x86_64 and
# aarch64, and macOS on Apple silicon (where tensorless can show recipes but serves no model).
set -eu

version="${TENSORLESS_VERSION:-0.2.0}"
dir="${TENSORLESS_INSTALL_DIR:-$HOME/.local/bin}"
repo="https://github.com/iammrduncan/tensorless"

say() { printf '%s\n' "$*"; }
fail() { printf 'tensorless install: %s\n' "$*" >&2; exit 1; }

case "$version" in
  '' | *[!0-9A-Za-z.-]*) fail "TENSORLESS_VERSION is not a version: '$version'" ;;
esac

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
  Linux/x86_64 | Linux/amd64) platform="linux-x86_64" ;;
  Linux/aarch64 | Linux/arm64) platform="linux-aarch64" ;;
  Darwin/arm64) platform="macos-arm64" ;;
  *) fail "no release archive for $os $arch. Build from source: go install github.com/iammrduncan/tensorless/cmd/tensorless@v$version" ;;
esac

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
else
  fail "needs curl or wget"
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  fail "needs sha256sum or shasum to check the download"
fi

name="tensorless-$version-$platform"
url="$repo/releases/download/v$version/$name.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading tensorless $version for $platform"
fetch "$url" "$tmp/$name.tar.gz" || fail "could not download $url"
fetch "$url.sha256" "$tmp/$name.tar.gz.sha256" || fail "could not download $url.sha256"

want="$(cut -d' ' -f1 <"$tmp/$name.tar.gz.sha256")"
got="$(sha256 "$tmp/$name.tar.gz")"
[ -n "$want" ] && [ "$want" = "$got" ] || fail "sha256 mismatch for $name.tar.gz: expected $want, got $got"
say "Checked sha256 $got"

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
[ -f "$tmp/$name/tensorless" ] || fail "the archive has no tensorless binary"
mkdir -p "$dir"
cp "$tmp/$name/tensorless" "$dir/tensorless.new"
chmod 755 "$dir/tensorless.new"
mv -f "$dir/tensorless.new" "$dir/tensorless"
say "Installed $dir/tensorless"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) say "" ; say "$dir is not on your PATH. Add it, for example:" ; say "  export PATH=\"$dir:\$PATH\"" ;;
esac

say ""
say "Next: check this machine, then run a model."
say "  tensorless doctor"
say "  tensorless run iamMrDuncan/Qwen3.8-27B-PN3-GGUF"
