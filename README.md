# SAT Training Labs

Job별 실습 코드와 Dockerfile입니다. 모델과 데이터는 실행 시 별도로 마운트합니다.

- [01-demo-main](labs/01-demo-main/) → ghcr.io/scpv2/sat-training-labs:demo-main-cuda
- [02-ynat](labs/02-ynat/) → ghcr.io/scpv2/sat-training-labs:ynat-cuda
- [03-fashion](labs/03-fashion/) → ghcr.io/scpv2/sat-training-labs:fashion-cuda
- [04-llm03](labs/04-llm03/) → ghcr.io/scpv2/sat-training-labs:llm03-cuda
- [05-ynat-resume](labs/05-ynat-resume/) → ghcr.io/scpv2/sat-training-labs:ynat-resume-cuda

## 데모 실행 가이드

https://sat.cloudsecuritylab.co.kr/local 에서 로그인 후 Training Job 생성을 엽니다. Spot Training, GPU 리소스 1개, PyTorch, 노드 1개를 선택합니다. 아래 이미지와 명령을 입력합니다. 각 Job의 완료와 결과 파일을 확인한 뒤 다음 Job을 생성합니다. 모델·데이터는 데모에 연결된 기존 S3 입력에서 가져옵니다. GPU EC2에 로컬 NVMe가 마운트되어 있으면 임시 작업 공간으로 사용하며, 준비되지 않은 환경에서는 EBS 경로를 사용합니다. 체크포인트와 최종 결과는 S3에 저장합니다. 데모의 실제 실행 장치는 AWS GPU EC2입니다.

### 01. demo-main

이미지 ghcr.io/scpv2/sat-training-labs:demo-main-cuda

명령

~~~bash
python /work/train.py --max-steps 10 --eval-samples 20
~~~

### 02. ynat

이미지 ghcr.io/scpv2/sat-training-labs:ynat-cuda

명령

~~~bash
python /lesson/train.py --task text --bundle /bundle --model /model --output /output --steps 5 --train-count 100 --eval-count 10 --checkpoint-steps 5
~~~

### 03. fashion

이미지 ghcr.io/scpv2/sat-training-labs:fashion-cuda

명령

~~~bash
python /lesson/train.py --task image --bundle /bundle --model /model --output /output --steps 5 --train-count 100 --eval-count 10 --checkpoint-steps 5
~~~

### 04. llm03

이미지 ghcr.io/scpv2/sat-training-labs:llm03-cuda

명령

~~~bash
python /lesson/train_llm03.py --model /model --dataset /dataset/train.jsonl --output /output/llm03 --steps 5 --checkpoint-steps 5
~~~

### 05. ynat-resume

이미지 ghcr.io/scpv2/sat-training-labs:ynat-resume-cuda

명령

~~~bash
python /lesson/checkpoint_resume_case.py --steps 6 --checkpoint-steps 3 --train-count 100 --eval-count 10
~~~

## 2026-11-24 실행 검증

5개 Job을 순차적으로 짧게 실행하고 GPU 학습, S3 결과 저장, 데모의 결과 가져오기를 확인했습니다.

| Job | 상태 | 가져온 결과 파일 |
|---|---|---:|
| [demo-main](https://sat.cloudsecuritylab.co.kr/local#jobs/3dc034e1ccc24ef980a77aff8b7655b4) | Succeeded | 17 |
| [ynat](https://sat.cloudsecuritylab.co.kr/local#jobs/3319c7e2cfd74c97918b32af44060ce9) | Succeeded | 12 |
| [fashion](https://sat.cloudsecuritylab.co.kr/local#jobs/7aadd0fd186145b1a52f6a565c8414eb) | Succeeded | 12 |
| [llm03](https://sat.cloudsecuritylab.co.kr/local#jobs/7fcd5084fb3a46f382599581781d1380) | Succeeded | 11 |
| [ynat-resume](https://sat.cloudsecuritylab.co.kr/local#jobs/f29b398a498c4e1a9e3c4283a267784d) | Succeeded | 13 |

ynat-resume은 S3에 3단계 체크포인트를 저장하고 첫 프로세스를 의도적으로 중단한 뒤, 새 프로세스에서 3단계를 복원해 6단계까지 완료했습니다.

입력 준비 시작부터 학습 전 평가 시작까지 관측된 시간은 EBS의 ynat에서 21분 50초, 로컬 NVMe의 ynat-resume 첫 프로세스에서 2분 13초였습니다. 같은 텍스트 입력을 사용한 서로 다른 Job의 관측값입니다.
