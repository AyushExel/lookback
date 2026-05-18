#!/usr/bin/env bash
# Build and publish the `lookback-ai` distribution to PyPI (or TestPyPI).
#
# Usage:
#   scripts/release.sh <version>                  # publish to PyPI
#   scripts/release.sh <version> --test           # publish to TestPyPI
#   scripts/release.sh <version> --dry-run        # build + verify, skip upload
#   scripts/release.sh <version> --allow-dirty    # skip the clean-git check
#
# Credentials:
#   - Set UV_PUBLISH_TOKEN to your PyPI API token (preferred).
#   - For TestPyPI runs, set TEST_PYPI_TOKEN instead (or UV_PUBLISH_TOKEN
#     if the test token is the only one configured).
#   - Tokens are shown once at creation: https://pypi.org/manage/account/token/
#     (TestPyPI: https://test.pypi.org/manage/account/token/).
#
# What it does, in order:
#   1. Parse args; verify <version> matches `pyproject.toml`.
#   2. Refuse to continue if git working tree is dirty (unless --allow-dirty).
#   3. Run the full test suite and ruff lint.
#   4. Clean `dist/` and run `uv build` (sdist + wheel).
#   5. `uv publish` (PyPI or TestPyPI based on flags).
#   6. Offer to git-tag the release and (optionally) push the tag.

set -euo pipefail

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
}

if [[ $# -lt 1 ]]; then
    usage
fi

VERSION="$1"
shift || true
TARGET="pypi"
DRY_RUN=0
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TARGET="testpypi" ;;
        --dry-run) DRY_RUN=1 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        -h|--help) usage ;;
        *)
            echo "unknown flag: $1" >&2
            usage
            ;;
    esac
    shift
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

bold "lookback-ai release  ·  version=${VERSION}  target=${TARGET}  dry_run=${DRY_RUN}"

# --- Step 1: version sanity ---------------------------------------------------
PYPROJECT_VERSION="$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/version = "(.+)"/\1/')"
if [[ "$PYPROJECT_VERSION" != "$VERSION" ]]; then
    die "version mismatch: argument=$VERSION but pyproject.toml=$PYPROJECT_VERSION. \
Bump pyproject.toml (and src/lookback/__init__.py) first, commit, then rerun."
fi
ok "version matches pyproject.toml ($VERSION)"

# --- Step 2: clean git --------------------------------------------------------
if [[ -d .git ]]; then
    if [[ $ALLOW_DIRTY -eq 0 ]]; then
        if ! git diff --quiet || ! git diff --cached --quiet; then
            die "git working tree is dirty. Commit or stash, or pass --allow-dirty."
        fi
        if [[ -n "$(git status --porcelain)" ]]; then
            die "untracked files present. Commit, gitignore, or pass --allow-dirty."
        fi
    fi
    ok "git tree clean"
else
    warn "not a git repository — skipping clean-tree check"
fi

# --- Step 3: tests + lint -----------------------------------------------------
bold "running tests"
uv run pytest -q
ok "tests passed"

bold "running ruff"
uv run ruff check src tests
ok "lint clean"

# --- Step 4: build ------------------------------------------------------------
bold "building artifacts"
rm -rf dist
uv build
ls dist/
ok "built sdist + wheel"

# --- Step 5: publish ----------------------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
    warn "dry run: skipping token check + upload"
    bold "artifacts ready in dist/  ·  target was: $TARGET"
    exit 0
fi

PUBLISH_ARGS=()
case "$TARGET" in
    pypi)
        TOKEN="${UV_PUBLISH_TOKEN:-${PYPI_TOKEN:-}}"
        if [[ -z "$TOKEN" ]]; then
            die "no PyPI token. Set UV_PUBLISH_TOKEN (preferred) or PYPI_TOKEN. \
Generate one at https://pypi.org/manage/account/token/"
        fi
        PUBLISH_ARGS+=(--token "$TOKEN")
        ;;
    testpypi)
        TOKEN="${TEST_PYPI_TOKEN:-${UV_PUBLISH_TOKEN:-}}"
        if [[ -z "$TOKEN" ]]; then
            die "no TestPyPI token. Set TEST_PYPI_TOKEN (preferred) or UV_PUBLISH_TOKEN. \
Generate one at https://test.pypi.org/manage/account/token/"
        fi
        PUBLISH_ARGS+=(--token "$TOKEN" --publish-url "https://test.pypi.org/legacy/")
        ;;
esac

bold "publishing to $TARGET"
uv publish "${PUBLISH_ARGS[@]}"
ok "published"

# --- Step 6: git tag ----------------------------------------------------------
if [[ -d .git ]]; then
    TAG="v${VERSION}"
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        warn "tag $TAG already exists locally — skipping tag step"
    else
        printf '\n'
        read -r -p "Create and push git tag $TAG? [y/N] " ANS
        if [[ "$ANS" =~ ^[Yy]$ ]]; then
            git tag -a "$TAG" -m "Release $VERSION"
            ok "tagged $TAG"
            read -r -p "Push tag to origin? [y/N] " ANS2
            if [[ "$ANS2" =~ ^[Yy]$ ]]; then
                git push origin "$TAG"
                ok "pushed $TAG"
            fi
        fi
    fi
fi

bold "release ${VERSION} → ${TARGET} complete"
case "$TARGET" in
    pypi)     echo "  https://pypi.org/project/lookback-ai/${VERSION}/" ;;
    testpypi) echo "  https://test.pypi.org/project/lookback-ai/${VERSION}/" ;;
esac
echo "Install:"
case "$TARGET" in
    pypi)     echo "  pip install lookback-ai==${VERSION}" ;;
    testpypi) echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lookback-ai==${VERSION}" ;;
esac
