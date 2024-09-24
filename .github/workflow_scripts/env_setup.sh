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

function setup_build_contrib_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
    python3 -m pip install awscli
    export AG_DOCS=1
}

function install_rag {
    python3 -m pip install --upgrade -e .
}
