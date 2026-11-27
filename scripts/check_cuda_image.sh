#!/bin/sh
# Builds the CUDA image with inert decoys planted in the build context, then checks the runtime
# ships only the virtual environment: no uv, checkout, weights, tokens or untracked package data.
set -eu
image=${1:-chriek-image-check}
cd "$(dirname "$0")/.."

# Top-level staging paths; the decoys are written only inside paths this run created.
staging="chriek-decoy.env chriek-decoy-hf chriek-decoy-weights src/chriek/chriek-decoy.env src/chriek/chriek_decoy"
decoys="chriek-decoy.env chriek-decoy-hf/token chriek-decoy-weights/model.safetensors
src/chriek/chriek-decoy.env src/chriek/chriek_decoy/model.safetensors"
collision=0
for path in $staging; do
    # -L also catches symlinks, including dangling ones that -e misses.
    if [ -e "$path" ] || [ -L "$path" ]; then
        echo "refusing to overwrite existing $path" >&2
        collision=1
    fi
done
[ "$collision" = 0 ] || exit 1
created=""
trap 'for path in $created; do rm -rf "$path"; done' EXIT
# noclobber: never write through a path that appeared after the check above.
set -C
for dir in chriek-decoy-hf chriek-decoy-weights src/chriek/chriek_decoy; do
    mkdir "$dir"
    created="$created $dir"
done
for decoy in $decoys; do
    echo "chriek decoy, not a secret" >"$decoy"
    created="$created $decoy"
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
