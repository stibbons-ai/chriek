#!/bin/sh
# Builds the CUDA image with inert decoys planted in the build context, then checks the runtime
# ships only the virtual environment: no uv, checkout, weights, tokens or untracked package data.
set -eu
image=${1:-chriek-image-check}
cd "$(dirname "$0")/.."

decoys="chriek-decoy.env chriek-decoy-hf/token chriek-decoy-weights/model.safetensors
src/chriek/chriek-decoy.env src/chriek/chriek_decoy/model.safetensors"
for decoy in $decoys; do
    if [ -e "$decoy" ]; then
        echo "refusing to overwrite existing $decoy" >&2
        exit 1
    fi
done
trap 'for decoy in $decoys; do rm -f "$decoy"; done; rmdir chriek-decoy-hf chriek-decoy-weights src/chriek/chriek_decoy 2>/dev/null || true' EXIT
for decoy in $decoys; do
    mkdir -p "$(dirname "$decoy")"
    echo "chriek decoy, not a secret" >"$decoy"
done

docker build -t "$image" .
# Root, so find can read every directory.
docker run --rm --user root --entrypoint sh "$image" -c '
set -eu
if [ "$(ls -A /app)" != .venv ]; then echo "/app holds more than .venv:"; ls -A /app; exit 1; fi
if command -v uv; then echo "uv is in the runtime image"; exit 1; fi
leaks=$(find / -xdev \( -iname "*decoy*" -o -name "*.safetensors" -o -name token \
    -o -name ".env" -o -name "*.env" -o -name .git \) -print)
if [ -n "$leaks" ]; then echo "unexpected files:"; echo "$leaks"; exit 1; fi
package=$(python -c "import chriek, pathlib; print(pathlib.Path(chriek.__file__).parent)")
data=$(find "$package" -type f ! -name "*.py" ! -name "*.pyc")
if [ -n "$data" ]; then echo "non-Python files in the package:"; echo "$data"; exit 1; fi
python -c "import boto3, diffusers, torch, chriek.cuda_backend"
chriek --help >/dev/null
echo "runtime image ok"
'
