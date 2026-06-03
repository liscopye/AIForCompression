#!/bin/bash
#SBATCH --job-name=dcvc_build
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_build_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_build_%j.log

set -euo pipefail

export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"
export CONDA_PREFIX=/data/run01/scxj523/zsh/envs/zsh

FM_ROOT=/data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/DCVC-family/DCVC-FM
FM_MODELS_DIR="$FM_ROOT/src/models"

# 1. 编译 DCVC-FM 熵编码扩展
echo "=== Building DCVC-FM entropy extension ==="
if find "$FM_MODELS_DIR" -maxdepth 1 -name 'MLCodec_rans*.so' | grep -q .; then
  echo "Using existing MLCodec_rans build in $FM_MODELS_DIR"
else
  cd "$FM_ROOT/src/cpp"
  rm -rf build/
  mkdir -p build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1
  make -j 2>&1
  cp -f "$FM_ROOT"/src/cpp/build/py_rans/MLCodec_rans*.so "$FM_MODELS_DIR"/
fi

# 2. 编译 DCVC-FM CUDA extension
echo "=== Building DCVC-FM CUDA extension ==="
cd "$FM_ROOT/src/models/extensions"
rm -rf build/ *.so
python3 setup.py build_ext --inplace 2>&1

# 3. 编译 DCVC-RT CUDA 推理扩展
echo "=== Building DCVC-RT CUDA inference extension (sm_120) ==="
RT_EXT_DIR=/data/run01/scxj523/zsh/project/AIForCompression/models/DCVC/src/layers/extensions/inference
cd "$RT_EXT_DIR"
rm -rf build/ *.so
python3 setup.py build_ext --inplace 2>&1

# 4. 验证编译结果
echo "=== Verifying builds ==="
find "$FM_ROOT" -name "*.so" -type f 2>/dev/null
find "$RT_EXT_DIR" -name "*.so" -type f 2>/dev/null

# 5. 运行 DCVC-FM 测试
echo "=== Running DCVC-FM test ==="
cd /data/run01/scxj523/zsh/project/AIForCompression
python -u scripts/test_video_intra_era5.py \
  --data_root "/data/run01/scxj523/zsh/project/Data/ERA5/2024" \
  --output_dir "/data/run01/scxj523/zsh/project/AIForCompression/logs/results/DCVC-FM" \
  --model DCVC_FM \
  --max_samples 1

# 6. 运行 DCVC-RT 测试
echo "=== Running DCVC-RT test ==="
python -u scripts/test_video_intra_era5.py \
  --data_root "/data/run01/scxj523/zsh/project/Data/ERA5/2024" \
  --output_dir "/data/run01/scxj523/zsh/project/AIForCompression/logs/results/DCVC-RT" \
  --model DCVC_RT \
  --max_samples 1
