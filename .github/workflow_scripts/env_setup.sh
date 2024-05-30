function setup_lint_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install "black>=22.3,<23.0"
    python3 -m pip install "isort>=5.10"
}

function install_rag_test {
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade -e "./[tests]"
    python3 -m pip install pytest
}