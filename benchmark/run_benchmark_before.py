#!/usr/bin/env python3
"""
run_check.py — C1/C2/C3 파이프라인 계측
실제 모델 파이프라인에서 변경 1·2·3의 실행 시간·GPU 메모리를 측정하고
check_results.log에 저장.  실행 마지막에 mean ± std 요약 출력.

사용법:
    cd /media/vilab/osh/vdemoire/base_before
    conda run -n vd python run_check.py
"""

import sys, os, logging, re, statistics, time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'check_results.log')
WARMUP   = 5    # GPU 캐시 안정화 (check=False, 미로깅)
REPS     = 10   # 측정 횟수 — 평균·표준편차 계산에 사용

# 입력 형상 (YML: gt_size=256, B=1, T=3, C=3)
B, T, C, H, W = 1, 3, 3, 256, 256

sys.path.insert(0, BASE_DIR)
from vd.utils import get_root_logger as _get_vd_logger


# ── [CHECK] 라인 수집기 ───────────────────────────────────────────────────────
class _CheckCapture(logging.Handler):
    """[CHECK] 라인을 메모리에 쌓아 두고 나중에 요약에 사용."""
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if '[CHECK]' in msg:
            self.lines.append(msg)


_cap = _CheckCapture()


# ── 로거 설정 ──────────────────────────────────────────────────────────────────
def _init_log():
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh  = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    # 모델 내부 로거 ([CHECK] 발신자, propagate=False)
    ml = _get_vd_logger()
    ml.addHandler(fh)
    ml.addHandler(_cap)   # 수집기 추가
    ml.setLevel(logging.INFO)

    # 스크립트 자체 메시지용 root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

_init_log()


# ── opt 구성 ───────────────────────────────────────────────────────────────────
def _build_opt():
    import yaml
    yml = os.path.join(BASE_DIR, 'options/train/Train_ipv2.yml')
    with open(yml) as f:
        opt = yaml.safe_load(f)

    opt['check']                            = True
    opt['is_train']                         = True
    opt['profile_flops']                    = False
    opt['dist']                             = False
    opt['num_gpu']                          = 1
    opt['logger']['wandb']['use_wandb']     = False
    opt['path']['pretrain_network']         = None
    opt['path']['resume_state']             = None
    opt['train']['ema_decay']               = 0      # EMA 없이 단순 비교
    opt['train']['perceptual_opt']          = None   # VGG 로딩 생략
    opt['train']['mid_perceptual_opt']      = None
    opt['train']['decomp_contrastive_opt']  = None
    return opt


# ── 더미 데이터 ────────────────────────────────────────────────────────────────
def _data(device):
    return {
        'lq':        torch.randn(B, T, C, H, W, device=device),
        'gt':        torch.randn(B,    C, H, W, device=device),
        'gt_frames': torch.randn(B, T, C, H, W, device=device),
    }


# ── 요약 출력 ─────────────────────────────────────────────────────────────────
def _summarize(capture, log, tag, c1_ms):
    """수집된 [CHECK] 라인을 파싱해 mean ± std 테이블 출력."""

    def _parse_method(keyword):
        """'time=Xms  peak=YMB' 형식 라인 파싱."""
        times, peaks = [], []
        for line in capture.lines:
            if keyword in line and 'time=' in line and '└─' not in line:
                t = re.search(r'time=\s*([\d.]+)ms', line)
                p = re.search(r'peak=\s*([\d.]+)MB', line)
                if t and p:
                    times.append(float(t.group(1)))
                    peaks.append(float(p.group(1)))
        return times, peaks

    def _parse_subop(keyword):
        """'└─ keyword: X ms' 형식 라인 파싱."""
        times = []
        for line in capture.lines:
            if '└─' in line and keyword in line:
                t = re.search(r':\s*([\d.]+)\s*ms', line)
                if t:
                    times.append(float(t.group(1)))
        return times

    def _fmt(vals):
        if not vals:
            return '   —'
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f'{m:7.2f} ± {s:5.2f}'

    c2_t, c2_p = _parse_method('C2')
    c3_t, c3_p = _parse_method('C3')
    ms_t        = _parse_subop('_match_stats')

    log.info('')
    log.info('━' * 68)
    log.info('[SUMMARY]  %s  (warmup=%d, reps=%d)', tag, WARMUP, REPS)
    log.info('━' * 68)
    log.info('%-36s %s  %s',
             '항목', '     time (ms, mean±std)', '  peak (MB, mean)')
    log.info('─' * 68)
    log.info('C1 __init__ (1회 측정, 워밍업 없음)  %s ms', f'{c1_ms:8.1f}')
    if c2_t:
        log.info('C2 optimize_parameters              %s  %7.1f',
                 _fmt(c2_t), statistics.mean(c2_p))
    if ms_t:
        log.info('   └─ _match_stats (C2 내 서브)    %s', _fmt(ms_t))
    if c3_t:
        log.info('C3 test()                           %s  %7.1f',
                 _fmt(c3_t), statistics.mean(c3_p))
    log.info('━' * 68)


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    log    = logging.getLogger()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    log.info('=' * 64)
    log.info('run_check.py  [BEFORE]  device=%s', device)
    log.info('  warmup=%d  reps=%d  shape=(%d,%d,%d,%d,%d)',
             WARMUP, REPS, B, T, C, H, W)
    log.info('  log → %s', LOG_FILE)
    log.info('=' * 64)

    # ── C1: __init__ ─────────────────────────────────────────────────────────
    log.info('')
    log.info('[PHASE] C1 — __init__  (check=True, 1회, 워밍업 없음)')
    opt = _build_opt()

    from vd.models.multiframevd_model import MultiFrameVDModel
    torch.cuda.synchronize()
    t0    = time.perf_counter()
    model = MultiFrameVDModel(opt)
    torch.cuda.synchronize()
    c1_ms = (time.perf_counter() - t0) * 1000
    log.info('  전체 __init__ 완료: %.1f ms', c1_ms)

    # ── 워밍업 (check=False) ──────────────────────────────────────────────────
    log.info('')
    log.info('[PHASE] 워밍업 (%d iter, check=False, 미로깅)', WARMUP)
    model._check = False
    for i in range(WARMUP):
        model.feed_data(_data(device))
        model.optimize_parameters(i)
    for _ in range(WARMUP):
        model.feed_data(_data(device))
        model.test()
    torch.cuda.synchronize()

    # ── C2: optimize_parameters ───────────────────────────────────────────────
    log.info('')
    log.info('[PHASE] C2 — optimize_parameters × %d (check=True)', REPS)
    model._check = True
    for i in range(REPS):
        model.feed_data(_data(device))
        model.optimize_parameters(WARMUP + i)

    # ── C3: test() ────────────────────────────────────────────────────────────
    log.info('')
    log.info('[PHASE] C3 — test() × %d (check=True)', REPS)
    for i in range(REPS):
        model.feed_data(_data(device))
        model.test()

    # ── 요약 ─────────────────────────────────────────────────────────────────
    _summarize(_cap, log, 'BEFORE', c1_ms)

    log.info('')
    log.info('완료 → %s', LOG_FILE)


if __name__ == '__main__':
    main()
