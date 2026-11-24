# CNN·CSV 학습 실습

Linux x86_64에서 PyTorch로 합성 이미지 CNN과 수치 CSV 분류를 실행합니다.
Docker 빌드 컨텍스트는 이 디렉터리입니다.

```bash
docker build --platform linux/amd64 -t sat-training-labs:cpu .
docker build --platform linux/amd64 --build-arg TORCH_CHANNEL=cu124 -t sat-training-labs:cu124 .
docker volume create sat-lab-results
docker run --rm -v sat-lab-results:/results sat-training-labs:cpu \
  python train.py --task synthetic_images --epochs 2 --device cpu --run-dir /results/cnn
docker run --rm -v sat-lab-results:/results sat-training-labs:cpu \
  python train.py --task csv --csv-file samples/dataset.csv --epochs 2 --device cpu --run-dir /results/csv
```

체크포인트와 학습 지표는 볼륨에 저장됩니다. 같은 경로에서 `--resume`과 더 큰 `--epochs`를 지정하면 이어서 학습합니다. CUDA는 NVIDIA Container Toolkit이 설치된 호스트에서 `docker run --gpus all`과 `--device cuda`를 사용합니다.

GHCR 게시 워크플로가 성공하면 `ghcr.io/scpv2/sat-training-labs:cpu`, `:cu124`를 같은 명령에 사용할 수 있습니다. 비공개 이미지에는 패키지 읽기 권한과 GHCR 인증이 필요합니다. SCP Simple AI Training에서 직접 가져올 수 있는지는 외부 레지스트리 입력·인증 지원과 네트워크 경로를 확인해야 합니다.
