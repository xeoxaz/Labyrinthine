# Arch Packaging Notes

This directory contains the Arch/CachyOS packaging scaffold for Labyrinthine.

## Files

- `PKGBUILD`: AUR-ready git package for `labyrinthine-git`
- `.SRCINFO`: metadata generated from `PKGBUILD`
- `PKGBUILD.stable`: template for a future tagged-release package

## Local validation

From the repository root:

```bash
./scripts/check-release-readiness.sh
```

To refresh `.SRCINFO` after validation and prepare the AUR metadata in one step:

```bash
./scripts/prepare-aur-package.sh
```

From this directory, if you specifically want to test the Arch package build against the published upstream GitHub repo:

```bash
makepkg -f
```

To install locally after a successful build:

```bash
makepkg -si
```

## Publishing `labyrinthine-git` to AUR

1. Make sure the upstream GitHub repository contains the code you want packaged.
2. Regenerate `.SRCINFO`:

```bash
makepkg --printsrcinfo > .SRCINFO
```

3. Create an AUR repository named `labyrinthine-git`.
4. Copy `PKGBUILD` and `.SRCINFO` into that AUR repository.
5. Commit and push to AUR.

## Publishing stable `labyrinthine`

Do this only after creating an upstream tag such as `v0.1.0`.

1. Copy `PKGBUILD.stable` to `PKGBUILD` in a clean packaging workspace for the stable AUR repo.
2. Set `pkgver` to the tag version without the leading `v`.
3. Update the source URL if the release tag naming scheme changes.
4. Generate the checksum:

```bash
updpkgsums
```

5. Regenerate `.SRCINFO`:

```bash
makepkg --printsrcinfo > .SRCINFO
```

6. Publish to an AUR repository named `labyrinthine`.

## Remaining blockers

- Push current upstream changes before publishing the git package