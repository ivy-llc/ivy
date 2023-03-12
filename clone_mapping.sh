USER_EMAIL="rashul.chutani@gmail.com"
USER_NAME="Rashul Chutani"
TARGET_BRANCH=$1
CLONE_DIR=$(mktemp -d)
GITHUB_SERVER="github.com"
mkdir --parents "$HOME/.ssh"
DEPLOY_KEY_FILE="$HOME/.ssh/deploy_key"
echo "${SSH_DEPLOY_KEY}" > "$DEPLOY_KEY_FILE"
chmod 600 "$DEPLOY_KEY_FILE"

SSH_KNOWN_HOSTS_FILE="$HOME/.ssh/known_hosts"
ssh-keyscan -H "$GITHUB_SERVER" > "$SSH_KNOWN_HOSTS_FILE"

export GIT_SSH_COMMAND="ssh -i "$DEPLOY_KEY_FILE" -o UserKnownHostsFile=$SSH_KNOWN_HOSTS_FILE"

# Setup git
git config --global user.email "$USER_EMAIL"
git config --global user.name "$USER_NAME"

git clone --single-branch --depth 1 --branch "$TARGET_BRANCH" git@github.com:unifyai/Mapping.git

