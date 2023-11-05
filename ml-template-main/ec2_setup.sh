#!/usr/bin/env bash

# Kill script on first error
set -o errexit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

TERMINAL_RESTART_REQUIRED="false"

# Install pyenv and build dependencies if required
if ! command -v pyenv &>/dev/null; then
    echo "pyenv not found on PATH. Attempting to install"
    # See: https://github.com/pyenv/pyenv#getting-pyenv
    git clone https://github.com/yyuu/pyenv.git ~/.pyenv

    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    TERMINAL_RESTART_REQUIRED="true"

    # Remove openssl-devel library, as this is openssl 1.0, and is incompatible with Python 3.10+
    sudo yum remove openssl-devel -y
    # Install build dependencies as per: https://github.com/pyenv/pyenv/wiki#suggested-build-environment                                                                                                                                                                                                            1 ↵
    sudo yum install gcc make patch zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl11-devel tk-devel libffi-devel xz-devel -y

    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
fi

export PYTHON_BIN=$(which python3)

# Log out python path
echo "PYTHON_BIN=${PYTHON_BIN}"
${PYTHON_BIN} --version

# Check that PATH is set up correctly for pipx installed executables
if ! command -v pipx &>/dev/null; then
    echo "pipx not found on PATH. Attempting to install"
    # Update pip
    ${PYTHON_BIN} -m pip install --upgrade pip

    # Install pipx for isolated poetry install
    ${PYTHON_BIN} -m pip install --user pipx
    ${PYTHON_BIN} -m pipx ensurepath

    # Add ~/.local/bin to the current PATH if it's not there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
      echo "Adding $HOME/.local/bin to PATH"
      export PATH=$HOME/.local/bin:$PATH

      echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.zshrc
      TERMINAL_RESTART_REQUIRED="true"
    fi
fi

# Install pipenv
if ! command -v pipenv &>/dev/null; then
    echo "pipenv not found on PATH. Attempting to install"
    pipx install pipenv==2023.6.26
fi

# Install ds cli
if ! command -v ds-cli &>/dev/null; then
    echo "ds-cli not found on PATH. Attempting to install"
    pipx install lit-ds-cli
fi

# Change ownership of mlflow artifacts directory
if [ -d "/mnt/mlflow-artifacts" ]; then
  sudo chown $(whoami):wheel /mnt/mlflow-artifacts
fi

if [ "${TERMINAL_RESTART_REQUIRED}" = "true" ]; then
    echo "*** Changes have been made which require a restart of the terminal. ***"
    echo "*** Please close this terminal window and open another to continue"
fi