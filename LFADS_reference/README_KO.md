# LFADS 데이터 전처리 코드 사용 설명서 (한국어)

> **대상:** IBL Brain-Wide Map의 M1(MOp) 데이터 및 기타 신경 spike 데이터  
> **목표:** Spike 데이터 → trial별 spike count → 학습/검증 분리 → LFADS 입력용 HDF5 저장  
> **현재 범위:** 데이터 전처리·검증과 PCA 미리보기까지. **LFADS 모델 학습 및 latent factor 추출은 아직 완료·검증되지 않음.**

이 폴더에는 기존 IBL M1 / LFADS 실습에서 사용했던 작업을 **재사용 가능한 독립 실행형 Python 스크립트**로 정리했습니다.

- `scripts/*.py`는 **Python 파일**입니다. 파일 안에 터미널 명령을 붙여넣지 마세요.
- 아래 코드 블록 중 `bash`로 표시한 명령은 **VS Code 터미널**에서 실행합니다.
- `06_pca_preview.py`는 **LFADS 결과가 아니라**, spike count로 계산한 발화율에 PCA를 적용하는 사전 분석입니다.

## 0. 폴더 구조

```text
LFADS_reference/
├── README.md                         # 영문 설명서
├── README_KO.md                      # 한글 설명서 (현재 파일)
├── requirements-preprocess.txt       # 전처리 의존성 목록
├── LFADS_DATAMODULE_TEMPLATE.yaml    # LFADS용 설정 참고 템플릿 (학습 검증 전)
├── scripts/
│   ├── lfads_data_utils.py           # 공통 함수: binning / 선별 / 분리 / 저장
│   ├── 01_find_ibl_insertions.py     # IBL 데이터에서 probe PID 검색
│   ├── 02_ibl_to_h5.py               # IBL PID → spike count → LFADS HDF5
│   ├── 03_other_data_to_h5.py        # IBL 이외의 .npz → LFADS HDF5
│   ├── 04_inspect_h5.py              # HDF5 데이터 검사
│   ├── 05_check_gpu.py               # NVIDIA / CUDA / PyTorch / Slurm 확인
│   └── 06_pca_preview.py             # 발화율 기반 PCA 시각화 (LFADS 아님)
├── examples/
│   └── make_example_npz.py           # 시험용 합성 데이터 생성
└── tests/
    └── test_pipeline.py             # 전처리 기능 단위 테스트
```

## 1. Python 환경 준비

### 1.1. 전처리용 가상환경

IBL 데이터 전처리에는 `one.api`, `brainbox.io.one`, `numpy`, `h5py` 등이 필요합니다. 기존 실습에서 사용한 환경은 `LFADS_pra/.venv`였습니다.

**VS Code 터미널:**

```bash
cd /home/changhyun/LFADS_pra
source .venv/bin/activate
python -m pip install h5py numpy matplotlib
```

IBL 관련 모듈이 없다면 **전처리용 가상환경에서만** 다음을 실행하세요.

```bash
python -m pip install -r LFADS_reference/requirements-preprocess.txt
```

`requirements-preprocess.txt`는 패키지 목록이며, 버전을 고정한 재현성 보장 파일(lockfile)은 아닙니다. 정상적으로 작동하는 환경이라면 불필요하게 패키지를 다시 설치하지 않는 것이 좋습니다.

> **Jupyter Notebook 사용 시:** 새 셀의 `%pip install h5py`는 현재 노트북 커널 환경에 설치합니다. 반면 터미널의 `pip`는 다른 Python을 가리킬 수 있습니다. 실제 사용 중인 Python 경로는 아래 명령으로 확인합니다.
>
> ```bash
> python -c 'import sys; print(sys.executable)'
> ```

### 1.2. GPU는 전처리에 필수가 아님

`01`~`04` 스크립트의 데이터 준비·저장 작업은 CPU에서 실행할 수 있습니다. **GPU가 필요한 것은 주로 이후 LFADS 모델 학습**입니다.

## 2. IBL 데이터 분석 과정: 새 PID에도 재사용

### 2.1. 분석 대상 PID 검색

`LFADS_reference/` 폴더로 이동한 상태에서 실행합니다.

```bash
cd /home/changhyun/LFADS_pra/LFADS_reference
python scripts/01_find_ibl_insertions.py --region MOp --limit 15
```

- `MOp`: 마우스 일차 운동피질(primary motor cortex) 관련 뇌 영역 약어입니다.
- 출력된 PID는 **후보 probe insertion ID**입니다.
- 특정 PID를 검색했다고 해서 MOp의 품질 좋은 뉴런이 많이 나온다는 보장은 없습니다.

### 2.2. 특정 PID의 spike 데이터를 HDF5로 변환

```bash
python scripts/02_ibl_to_h5.py \
  --pid 0ffa052a-fe3f-46c2-999d-43bd56b7826e \
  --region MOp \
  --quality good \
  --event firstMovement_times \
  --before 0.5 --after 1.0 --bin-size 0.010 \
  --out-dir lfads_data
```

**스크립트 내부 처리 순서**

1. `ONE(base_url='https://openalyx.internationalbrainlab.org')`으로 IBL 서버에 연결합니다.
2. `SpikeSortingLoader`로 spike 시점과 cluster/channel 정보를 불러옵니다.
3. 지정한 뇌 영역(`MOp`)과 good unit(`clusters['label'] == 1`)을 선별합니다. Spike의 cluster ID는 배열 행 번호가 아니라 **실제 `clusters['cluster_id']` 값**을 사용합니다.
4. `trials['firstMovement_times']`에서 `NaN` 등 유효하지 않은 이벤트를 제외합니다. 이 필터를 **spike count 생성 전에 한 번만 적용**하여 `X`, `event_times`, 원본 `trial_ids`의 대응을 유지합니다.
5. 각 trial의 이벤트 시각을 기준으로 -0.5초~+1.0초 동안 뉴런별 spike 수를 **10 ms 간격으로 계산**합니다.
6. 실제 이벤트 시간 순으로 trial을 정렬하여 대략 **80% 학습 / 20% 검증**으로 나눕니다. 검증 구간과 겹치는 학습 window는 제외합니다.
7. 결과와 메타데이터를 HDF5(`.h5`)로 저장합니다.

**데이터 배열의 의미**

```text
X.shape = (trial 수, time bin 수, neuron 수)
```

여기서 `X`의 값은 **발화율(Hz)이 아니라 정수 spike count**입니다.

### 2.3. 새로운 데이터에서 변경할 옵션

| 옵션 | 설명 |
|---|---|
| `--pid` | 분석할 새 probe insertion UUID (**필수**) |
| `--region` | 대상 영역 약어 접두어(`MOp`) 또는 `ALL`. 영역을 넓히면 뉴런 수는 늘 수 있지만 해석 대상도 달라짐 |
| `--quality` | `good` 또는 `all`. 원칙적으로 품질 좋은 unit 우선 |
| `--event` | `trials`의 시간 필드, 예: `firstMovement_times` (해당 데이터에 존재해야 함) |
| `--before`, `--after` | 이벤트 전후 분석 구간(단위: 초) |
| `--bin-size` | Bin 폭(단위: 초), 예: `0.010` = 10 ms |
| `--revision latest` | 기존에 지정한 spike sorting revision을 사용할 수 없을 때 최신 revision 시도 |
| `--out-dir` | HDF5 파일 저장 폴더. 기본값은 명령을 실행한 위치의 `lfads_data/` |
| `--overwrite` | 기존 HDF5 덮어쓰기. **의도할 때만 사용** |

분석 구간이 이벤트 전 0.5초 + 후 1.0초이고 bin이 10 ms이면 **trial당 150개 time bin**입니다.

`--event`는 데이터를 정렬할 행동 시점입니다. **LFADS 모델의 초기 잠재 상태 `g₀`는 선택한 trial 분석 구간의 시작 상태**이며, 이벤트 시점과 반드시 같지는 않습니다.

### 2.4. 기존 M1 실습에서 확인한 결과

```text
유효한 이벤트 trial 수: 653
원본 X.shape:         (653, 150, 5)
Train shape:         (522, 150, 5)
Validation shape:    (131, 150, 5)
겹쳐서 제외한 trial:   0
```

이 기록에서는 MOp good unit이 **5개**만 선택됐습니다. 전처리 및 학습 동작 테스트에는 사용 가능하지만, 운동피질 **신경집단 동역학을 신뢰성 있게 해석하기에는 적은 수**입니다. 후속 연구에서는 더 많은 뉴런이 확보되는 PID를 찾는 것이 좋습니다.

## 3. IBL이 아닌 다른 데이터셋 적용 (`.npz` 입력)

`03_other_data_to_h5.py`는 **두 가지 NumPy `.npz` 형식**을 지원합니다. 이벤트 시각과 spike 시각은 **동일한 기록 시간축**을 사용해야 하며, 시간 단위는 **초(s)**여야 합니다.

### 형식 A: 이미 binning한 spike count

**Python `.py` 파일 또는 Jupyter 셀:**

```python
import numpy as np

np.savez_compressed(
    'my_binned_dataset.npz',
    X=X,                       # (n_trials, n_bins, n_units), 정수 spike counts
    event_times=event_times,   # (n_trials,), X와 trial별로 대응하는 절대 시각
    cluster_ids=cluster_ids,   # (n_units,), 선택 사항
    trial_ids=trial_ids,       # (n_trials,), 선택 사항
    bin_edges=bin_edges,       # (n_bins+1,), 이벤트 상대 시각; 선택 사항
)
```

### 형식 B: binning 전 원시 spike 시각

**Python `.py` 파일 또는 Jupyter 셀:**

```python
import numpy as np

np.savez_compressed(
    'my_raw_dataset.npz',
    spike_times=spike_times,       # (n_spikes,), 초 단위 spike 시각
    spike_clusters=spike_clusters, # (n_spikes,), spike별 뉴런/cluster ID
    event_times=event_times,       # (n_trials,), 초 단위 이벤트 시각
    cluster_ids=cluster_ids,       # 분석할 뉴런 ID, 선택 사항
)
```

변환 명령은 다음과 같습니다. **VS Code 터미널:**

```bash
python scripts/03_other_data_to_h5.py \
  --input my_raw_dataset.npz \
  --output lfads_data/other_dataset.h5 \
  --before 0.5 --after 1.0 --bin-size 0.010
```

이미 binning한 데이터는 `.npz` 파일에 원래 `bin_edges`를 포함하는 편이 좋습니다. 포함하지 않는다면 CLI의 `--before`, `--after`, `--bin-size`가 기존 binning 설정과 **정확히 일치**해야 합니다.

> **주의:** 형식만 맞다고 trial과 이벤트가 실제로 올바르게 대응한다는 보장은 없습니다. 원본 데이터에서 trial 정렬이 맞는지 반드시 확인해야 합니다.

## 4. 저장된 HDF5 검증과 PCA 미리보기

**VS Code 터미널 (`LFADS_reference/` 내부):**

```bash
python scripts/04_inspect_h5.py lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5
python scripts/06_pca_preview.py lfads_data/m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5
```

LFADS-torch용으로 저장한 HDF5에는 다음 **4개 핵심 데이터셋**이 포함됩니다.

```text
train_encod_data   (train_trials, time_bins, units)
train_recon_data   (train_trials, time_bins, units)
valid_encod_data   (valid_trials, time_bins, units)
valid_recon_data   (valid_trials, time_bins, units)
```

기본 구성에서는 encoder 입력과 reconstruction target에 **동일한 spike count**를 사용합니다. 이외에 다음 정보도 저장됩니다.

- `train_trial_ids`, `valid_trial_ids`: 원본 trial ID
- `train_event_times_s`, `valid_event_times_s`: 각 trial의 정렬 이벤트 시각(초)
- `cluster_ids`: 선택한 뉴런의 실제 ID
- `bin_edges_s`: 이벤트 상대 bin 경계 시각
- 기타 PID, EID, bin 크기, 분리 방식 관련 메타데이터

**PCA와 LFADS의 차이**

- `06_pca_preview.py`: trial 평균 **발화율**에 선형 PCA를 적용하여 시간에 따른 궤적을 시각화
- **LFADS**: 반복 trial을 이용해 **개별 trial의 잠재 신경 동역학**을 추정하는 별도의 모델

따라서 `06_pca_preview.py`의 결과를 **LFADS latent trajectory라고 해석하면 안 됩니다.** 특히 뉴런이 5개라면 PCA 해석에도 주의해야 합니다.

## 5. 연구실 공용 GPU / Slurm 사용법

IBL 데이터 전처리만 수행한다면 GPU가 없어도 됩니다. **LFADS 모델 학습**을 GPU로 수행할 때 공용 GPU 자원을 할당받습니다.

기존 실습 당시 `neuroai4`에는 Slurm에 등록된 GPU가 없었으며 `neuroai1`, `neuroai2`, `neuroai3`, `neuroai5`에 GPU가 등록되어 있었습니다. **현재 배치 상태는 언제든 바뀔 수 있습니다.**

**VS Code 터미널:**

```bash
# GPU 노드 / 대기 작업 확인
sinfo -N -o '%N %P %t %G'
squeue -u "$USER"

# GPU 1개를 최대 30분간 할당받아 대화형 셸 실행
srun --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --time=00:30:00 --pty bash

# 할당받은 GPU 노드의 셸에서 실행
conda activate lfads-torch
python scripts/05_check_gpu.py

# 사용 종료 후 GPU 할당 해제
exit
```

`05_check_gpu.py`는 GPU 정보, PyTorch/CUDA 인식 여부, 간단한 CUDA 텐서 연산 등을 점검합니다.

**기존 실습에서 확인한 환경:**

```text
IBL 전처리 환경: /home/changhyun/LFADS_pra/.venv, Python 3.10.12
GPU 확인 노드:    neuroai5
GPU 모델:         NVIDIA GeForce RTX 4090
초기 .venv Torch: CPU 전용 빌드
LFADS 별도 환경:  torch 1.13.1+cu117, GPU 할당 후 CUDA 사용 성공
```

`nvidia-smi`가 실패했다고 해서 연구실 공용 서버에 NVIDIA 드라이버를 임의로 설치하거나 재부팅하면 안 됩니다. 우선 **GPU가 할당된 노드인지** 확인하고, 서버 문제라면 관리자에게 문의하세요.

## 6. LFADS-torch 설치 및 모델 학습: 아직 다음 단계

- LFADS-torch 공식 저장소: https://github.com/arsedler9/lfads-torch
- IBL spike sorting 데이터 로딩 참고: https://docs.internationalbrainlab.org/notebooks_external/loading_spikesorting_data.html

기존 IBL Notebook 환경을 보존하기 위해 LFADS 학습에는 **별도의 conda 환경**을 사용하도록 계획했습니다.

**VS Code 터미널:**

```bash
conda activate lfads-torch
cd /home/changhyun/LFADS_pra
git clone https://github.com/arsedler9/lfads-torch.git
python -m pip install -e ./lfads-torch
```

이미 `lfads-torch/` 폴더가 있다면 `git clone`은 생략해야 합니다. 위 명령은 **설치 참고용**이며, 설치 성공·전체 의존성 호환성은 별도로 확인해야 합니다.

**실제 모델 학습을 시작하려면** LFADS-torch 저장소에서 모델 설정과 datamodule 설정을 만들어야 합니다. 이때 입력 뉴런 수(기존 데이터는 5), time bin 수(150), 데이터 경로 등이 일치해야 합니다.

> **현재 프로젝트에서 학습 완료 모델과 학습 설정은 아직 검증되지 않았습니다.** `LFADS_DATAMODULE_TEMPLATE.yaml`은 참고 템플릿이지, 그대로 실행하면 학습이 완료되는 설정이라고 보장하지 않습니다.

## 7. 자주 발생한 오류와 해결법

| 오류 또는 증상 | 원인 / 조치 |
|---|---|
| `ModuleNotFoundError: No module named 'h5py'` | 실행 중인 Python 환경에 `h5py` 설치. Jupyter라면 `%pip install h5py` 사용 |
| `X`, `event_times`, `valid_trials` 개수 불일치 | 유효 이벤트 필터를 `X` 생성 **전에** 적용해야 함. `02_ibl_to_h5.py`는 이 과정을 일관되게 처리 |
| Cluster ID와 배열 행 번호 혼동 | `np.where(good_mask)[0]`를 실제 spike cluster ID로 사용하면 안 됨. `clusters['cluster_id'][good_mask]` 사용 |
| `FileExistsError` | 같은 이름의 `.h5` 파일이 이미 있음. 새 이름 사용 또는 의도한 경우에만 `--overwrite` |
| `nvidia-smi` 실행 실패 | 접속 노드와 Slurm GPU 할당 여부 확인 |
| `torch.cuda.is_available() == False` | GPU 할당, 사용 중인 Python 경로, CPU 전용 PyTorch(`+cpu`) 여부 확인 |
| MOp good unit이 5개 등 매우 적음 | 다른 PID를 확인하거나 분석 조건 재검토. 다른 영역까지 합쳤다면 M1만의 결과라고 해석하지 말 것 |

## 8. IBL 다운로드 없이 전처리 흐름 시험하기

**VS Code 터미널 (`LFADS_reference/` 내부):**

```bash
python examples/make_example_npz.py
python scripts/03_other_data_to_h5.py --input examples/demo_raw_spikes.npz --output lfads_data/demo_raw_spikes.h5
python scripts/04_inspect_h5.py lfads_data/demo_raw_spikes.h5
```

여기서 사용하는 데이터는 **합성 예제 데이터**입니다. 전처리 코드가 작동하는지 확인하는 용도이며 신경과학적 결론을 낼 수 있는 데이터가 아닙니다.

## 9. 단위 테스트

**VS Code 터미널 (`LFADS_reference/` 내부):**

```bash
python -m unittest discover -s tests -v
```

다음 항목을 검사합니다.

1. 배열 행 번호가 아닌 실제 cluster ID가 선택되는지
2. Spike binning 결과가 `numpy.histogram`과 일치하는지
3. 유효 이벤트 필터링 이후 원본 trial ID가 보존되는지
4. HDF5 배열의 shape과 학습/검증 window 겹침 제거가 올바른지

## 10. 새 데이터 분석 시 최소 실행 순서

**IBL 데이터인 경우:**

```bash
cd /home/changhyun/LFADS_pra/LFADS_reference

# 1) 새 PID 탐색
python scripts/01_find_ibl_insertions.py --region MOp --limit 15

# 2) <NEW_PID>를 실제 UUID로 교체하여 HDF5 생성
python scripts/02_ibl_to_h5.py \
  --pid <NEW_PID> \
  --region MOp --quality good \
  --event firstMovement_times \
  --before 0.5 --after 1.0 --bin-size 0.010 \
  --out-dir lfads_data

# 3) 위 명령에서 출력된 실제 파일명을 사용해 검사
python scripts/04_inspect_h5.py lfads_data/<SAVED_FILENAME>.h5
```

`<NEW_PID>`와 `<SAVED_FILENAME>`은 **설명용 자리표시자**입니다. 실제 값으로 바꾸지 않으면 명령이 실행되지 않습니다.

**마지막으로:** HDF5 저장까지는 **LFADS 학습 데이터 준비 단계**입니다. 이후 *모델 학습 → latent factors 추출 → trial별 동역학 비교 → (필요하다면) PCA 시각화*는 별도의 분석 단계로 진행해야 합니다.
