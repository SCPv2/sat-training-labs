# SAT Training Labs

Job별 실습 코드와 Dockerfile입니다. 모델과 데이터는 실행 시 별도로 마운트합니다.

- demo-main → ghcr.io/scpv2/sat-training-labs:demo-main-cuda
- ynat → ghcr.io/scpv2/sat-training-labs:ynat-cuda
- fashion → ghcr.io/scpv2/sat-training-labs:fashion-cuda
- llm03 → ghcr.io/scpv2/sat-training-labs:llm03-cuda
- ynat-resume → ghcr.io/scpv2/sat-training-labs:ynat-resume-cuda

## 데모 실행 가이드

https://sat.cloudsecuritylab.co.kr/local 에서 로그인 후 Training Job 생성을 엽니다. Spot Training, GPU 리소스 1개, PyTorch, 노드 1개를 선택합니다. 아래 이미지와 명령을 입력합니다. 각 Job의 완료와 결과 파일을 확인한 뒤 다음 Job을 생성합니다. 데모의 실제 실행 장치는 AWS GPU EC2입니다.

### demo-main

이미지 ghcr.io/scpv2/sat-training-labs:demo-main-cuda

명령

~~~bash
python /work/train.py --max-steps 10 --eval-samples 20
~~~

### ynat

이미지 ghcr.io/scpv2/sat-training-labs:ynat-cuda

명령

~~~bash
python /lesson/train.py --task text --bundle /bundle --model /model --output /output --steps 5 --train-count 100 --eval-count 10 --checkpoint-steps 5
~~~

### fashion

이미지 ghcr.io/scpv2/sat-training-labs:fashion-cuda

명령

~~~bash
python /lesson/train.py --task image --bundle /bundle --model /model --output /output --steps 5 --train-count 100 --eval-count 10 --checkpoint-steps 5
~~~

### llm03

이미지 ghcr.io/scpv2/sat-training-labs:llm03-cuda

명령

~~~bash
python /lesson/train_llm03.py --model /model --dataset /dataset/train.jsonl --output /output/llm03 --steps 5 --checkpoint-steps 5
~~~

### ynat-resume

이미지 ghcr.io/scpv2/sat-training-labs:ynat-resume-cuda

명령

~~~bash
python /lesson/checkpoint_resume_case.py --steps 6 --checkpoint-steps 3 --train-count 100 --eval-count 10
~~~
