USER_EMAIL="rashul.chutani@gmail.com"
USER_NAME=""
TARGET_BRANCH=master
CLONE_DIR=$(mktemp -d)
SSH_DEPLOY_KEY=${1}
GITHUB_SERVER="github.com"
mkdir --parents "$HOME/.ssh"
DEPLOY_KEY_FILE="$HOME/.ssh/deploy_key_"
echo "${SSH_DEPLOY_KEY}" > "$DEPLOY_KEY_FILE"
chmod 600 "$DEPLOY_KEY_FILE"

SSH_KNOWN_HOSTS_FILE="$HOME/.ssh/known_hosts"
ssh-keyscan -H "$GITHUB_SERVER" > "$SSH_KNOWN_HOSTS_FILE"

export GIT_SSH_COMMAND="ssh -i "$DEPLOY_KEY_FILE" -o UserKnownHostsFile=$SSH_KNOWN_HOSTS_FILE"

echo "Hello!"

# Setup git
git config --global user.email "$USER_EMAIL"
git config --global user.name "$USER_NAME"

{
	git clone --single-branch --depth 1 --branch "$TARGET_BRANCH" git@github.com:unifyai/Mapping.git "$CLONE_DIR"
} || {
	echo "::error::Could not clone the destination repository. Command:"
	echo "::error::git clone --single-branch --branch $TARGET_BRANCH $GIT_CMD_REPOSITORY $CLONE_DIR"
	echo "::error::(Note that if they exist USER_NAME and API_TOKEN is redacted by GitHub)"
	echo "::error::Please verify that the target repository exist AND that it contains the destination branch name, and is accesible by the API_TOKEN_GITHUB OR SSH_DEPLOY_KEY"
	exit 1
}

