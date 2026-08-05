#!/usr/bin/env bash
set -Eeuo pipefail

# Upload the project as a brand-new snapshot. If the remote branch exists, its
# content and branch history are replaced by a single root commit.

DEFAULT_REPO="git@github.com:MurrayTom/ToolHazard.git"
DEFAULT_BRANCH="main"
REPO_URL="${GITHUB_REPO:-$DEFAULT_REPO}"
BRANCH="${GITHUB_BRANCH:-$DEFAULT_BRANCH}"
COMMIT_MESSAGE="${GITHUB_COMMIT_MESSAGE:-Upload project snapshot}"
ASSUME_YES=false

usage() {
  cat <<'EOF'
Usage: ./upload_github.sh [options]

Options:
  -r, --repo URL       GitHub repository URL
  -b, --branch NAME    Remote branch to replace (default: main)
  -m, --message TEXT   Commit message
  -y, --yes            Skip the overwrite confirmation
  -h, --help           Show this help

Environment variables:
  GITHUB_REPO, GITHUB_BRANCH, GITHUB_COMMIT_MESSAGE

Examples:
  ./upload_github.sh --repo git@github.com:USER/REPO.git
  ./upload_github.sh --message "Release snapshot" --yes
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

while (($# > 0)); do
  case "$1" in
    -r|--repo)
      (($# >= 2)) || die "$1 requires a value"
      REPO_URL="$2"
      shift 2
      ;;
    -b|--branch)
      (($# >= 2)) || die "$1 requires a value"
      BRANCH="$2"
      shift 2
      ;;
    -m|--message)
      (($# >= 2)) || die "$1 requires a value"
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

command -v git >/dev/null 2>&1 || die "git is not installed"
git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 || die "invalid branch name: $BRANCH"

# Always upload the directory containing this script, regardless of the caller's cwd.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_DIR"

if [[ ! -d .git ]]; then
  git init
  git symbolic-ref HEAD "refs/heads/$BRANCH"
  printf 'Initialized Git repository in %s\n' "$PROJECT_DIR"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git var GIT_AUTHOR_IDENT >/dev/null 2>&1 ||
  die 'Git identity is missing; run git config --global user.name and user.email first'

REMOTE_REF="refs/heads/$BRANCH"
REMOTE_COMMIT="$(git ls-remote --heads origin "$REMOTE_REF" | awk 'NR == 1 {print $1}')"

if [[ -n "$REMOTE_COMMIT" ]]; then
  printf '\nWARNING: this will replace %s at:\n  %s\n' "$BRANCH" "$REPO_URL"
  printf 'The branch will contain only the current project snapshot and one new commit.\n'
  if [[ "$ASSUME_YES" != true ]]; then
    read -r -p 'Type OVERWRITE to continue: ' CONFIRMATION
    [[ "$CONFIRMATION" == "OVERWRITE" ]] || die "upload cancelled"
  fi
else
  printf 'Remote branch %s does not exist; performing first upload.\n' "$BRANCH"
fi

# Build a root commit with a temporary index. This respects .gitignore, includes
# new files, and excludes files deleted from the working tree.
TEMP_INDEX="$(mktemp "${TMPDIR:-/tmp}/toolhazard-upload-index.XXXXXX")"
trap 'rm -f "$TEMP_INDEX"' EXIT
rm -f "$TEMP_INDEX"
export GIT_INDEX_FILE="$TEMP_INDEX"
git read-tree --empty
git add --all -- .
TREE="$(git write-tree)"
NEW_COMMIT="$(printf '%s\n' "$COMMIT_MESSAGE" | git commit-tree "$TREE")"
unset GIT_INDEX_FILE

if [[ -n "$REMOTE_COMMIT" ]]; then
  git push --force-with-lease="$REMOTE_REF:$REMOTE_COMMIT" \
    origin "$NEW_COMMIT:$REMOTE_REF"
else
  git push origin "$NEW_COMMIT:$REMOTE_REF"
fi

# Keep the local branch aligned with the uploaded root commit without changing files.
git update-ref "refs/heads/$BRANCH" "$NEW_COMMIT"
git symbolic-ref HEAD "refs/heads/$BRANCH"
git reset --mixed --quiet "$NEW_COMMIT"
git branch --set-upstream-to="origin/$BRANCH" "$BRANCH" >/dev/null 2>&1 || true

printf '\nUpload complete.\nRepository: %s\nBranch:     %s\nCommit:     %s\n' \
  "$REPO_URL" "$BRANCH" "$NEW_COMMIT"
