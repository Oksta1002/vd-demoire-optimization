# Video Demoireing Model Optimization

BasicSR 기반 비디오 디모아레 파이프라인(`MultiFrameVDModel`)의 코드 최적화 과제 제출 저장소.

최적화 대상 파일: `vd/models/multiframevd_model.py` 단독.

---

## 최적화 요약

| # | 카테고리 | 대상 | 핵심 변경 |
|---|---------|------|---------|
| 1 | **C** (클래스 구조) | `__init__` + `_profile_flops()` | FLOPs 프로파일링 opt-in 분리 (`profile_flops` 플래그) |
| 2 | **C + E** (구조 + DL) | `optimize_parameters` + `_loss_step()` + `init_training_settings` | `_match_stats` 2→1회 공유, `gt_flat` view 4→1회 공유, 손실 누적 헬퍼 패턴화, perceptual 래퍼 |
| 3 | **D + E** (데코레이터 + DL) | `test()` | EMA/non-EMA 브랜치 중복 제거, `@_restore_train_mode` + `@torch.inference_mode()` 스택 |

### 정량 결과 (warmup=5, reps=10, shape=(1,3,3,256,256))

| 항목 | BEFORE | AFTER | 개선 |
|------|--------|-------|------|
| C1 `__init__` | 1870 ms | 435 ms | **−76.7 %** (프로파일링 제거) |
| C2 `optimize_parameters` | 80.1 ± 8.6 ms | 81.8 ± 8.0 ms | 동등 (측정 조건상 차이 없음, 아래 참고) |
| C3 `test()` | 24.7 ± 5.7 ms | 18.8 ± 1.6 ms | **−23.9 %, std −72 %** |

> **C2 차이 없음 이유**: 벤치마크에서 `decomp_contrastive_opt=None`으로 설정해 VGG 로딩을 생략했기 때문에,
> BEFORE에서도 두 번째 `_match_stats` 호출 블록이 실행되지 않음. 실제 학습(모든 손실 활성) 환경에서는
> `_match_stats` 1회 절감(~2.5 ms) 효과가 나타남.

---

## 프로젝트 구조

```
vd-demoire-optimization/
├── README.md
├── environment.yml          # conda 환경 설정
├── src/
│   ├── before/              # 최적화 전 전체 코드
│   │   ├── vd/models/multiframevd_model.py   ← 핵심 파일
│   │   ├── train.py
│   │   ├── test.py
│   │   └── options/
│   └── after/               # 최적화 후 전체 코드
│       ├── vd/models/multiframevd_model.py   ← 핵심 파일
│       ├── train.py
│       ├── test.py
│       └── options/
├── benchmark/
│   ├── run_benchmark_before.py   # BEFORE 파이프라인 계측 스크립트
│   └── run_benchmark_after.py    # AFTER 파이프라인 계측 스크립트
└── results/
    ├── before_results.log         # BEFORE 원본 측정 로그
    ├── after_results.log          # AFTER 원본 측정 로그
    └── benchmark_results.csv      # 요약 비교 CSV
```

---

## 환경 설정

```bash
conda env create -f environment.yml
conda activate vd
```

> DeepSpeed의 DCN 커스텀 CUDA 커널 빌드가 필요합니다:
> ```bash
> cd src/before/vd/archs/dcn   # 또는 src/after/vd/archs/dcn
> python setup.py build_ext --inplace
> ```

---

## 벤치마크 실행

벤치마크 스크립트는 더미 텐서(synthetic data)로 실제 GPU 파이프라인을 측정합니다.
실제 데이터셋 없이도 실행 가능합니다.

### BEFORE 측정

```bash
cd src/before
conda run -n vd python ../../benchmark/run_benchmark_before.py
# 결과: src/before/check_results.log  (또는 benchmark 실행 디렉토리 내 생성)
```

### AFTER 측정

```bash
cd src/after
conda run -n vd python ../../benchmark/run_benchmark_after.py
# 결과: src/after/check_results.log
```

### 측정 항목

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `WARMUP` | 5 | GPU 캐시 안정화용 반복 (미로깅) |
| `REPS` | 10 | 실측 반복 횟수 |
| 입력 shape | (1,3,3,256,256) | B=1, T=3, C=3, H=W=256 |

출력 로그에는 각 반복의 `time=Xms  peak=YMB` 값과 마지막에 `mean ± std` 요약 테이블이 출력됩니다.

---

## 전후 코드 비교

핵심 변경 파일만 비교하려면:

```bash
diff src/before/vd/models/multiframevd_model.py \
     src/after/vd/models/multiframevd_model.py
```
