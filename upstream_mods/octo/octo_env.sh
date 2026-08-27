export LD_LIBRARY_PATH=$(ls -d ~/octo/.venv/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
