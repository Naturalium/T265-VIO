"""
VIO 학습 문서 PDF 생성기
Stereo Visual-Inertial Odometry (T265) 이론 + 의사코드
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np
import textwrap

# ── 폰트 설정 ─────────────────────────────────────────────────────────────────
FONT_PATH  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
MONO_PATH  = '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf'

font_manager.fontManager.addfont(FONT_PATH)
font_manager.fontManager.addfont(FONT_BOLD)
font_manager.fontManager.addfont(MONO_PATH)

KR   = font_manager.FontProperties(fname=FONT_PATH)
KR_B = font_manager.FontProperties(fname=FONT_BOLD)
MONO = font_manager.FontProperties(fname=MONO_PATH)

PAGE_W, PAGE_H = 8.27, 11.69   # A4 inches
MARGIN_L, MARGIN_R = 0.10, 0.97
CONTENT_TOP = 0.93
BLUE   = '#1565C0'
LBLUE  = '#E3F2FD'
GRAY   = '#424242'
LGRAY  = '#F5F5F5'
GREEN  = '#1B5E20'
LGREEN = '#E8F5E9'
ORANGE = '#E65100'
PURPLE = '#4A148C'
RED    = '#B71C1C'

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def new_page(pdf, title='', chapter=''):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # 헤더 배경
    ax.add_patch(patches.Rectangle((0, 0.955), 1, 0.045,
                                   color=BLUE, transform=ax.transAxes))
    ax.text(0.5, 0.968, 'Visual-Inertial Odometry — 이론과 구현',
            color='white', fontproperties=KR, fontsize=9,
            ha='center', va='center', transform=ax.transAxes)
    if chapter:
        ax.text(MARGIN_L, 0.968, chapter, color='#90CAF9',
                fontproperties=KR, fontsize=7.5, va='center',
                transform=ax.transAxes)

    # 타이틀
    if title:
        ax.text(MARGIN_L, 0.925, title, color=BLUE,
                fontproperties=KR_B, fontsize=15, va='top',
                transform=ax.transAxes)
        # 밑줄 — 제목 텍스트(fontsize=15 → ~0.018 axes fraction) 아래에 배치
        ax.add_patch(patches.Rectangle((MARGIN_L, 0.900), 0.87, 0.0015,
                                       color=BLUE, transform=ax.transAxes))

    return fig, ax


def txt(ax, x, y, s, fp=None, size=9.5, color='black', bold=False, ha='left', va='top'):
    fp = fp or (KR_B if bold else KR)
    ax.text(x, y, s, fontproperties=fp, fontsize=size,
            color=color, ha=ha, va=va, transform=ax.transAxes)


def math_eq(ax, x, y, latex, size=11, color='black', ha='center'):
    """matplotlib mathtext 수식 렌더링"""
    ax.text(x, y, latex, fontsize=size, color=color,
            ha=ha, va='top', transform=ax.transAxes,
            fontfamily='DejaVu Sans')


def section(ax, x, y, title, size=11):
    """절 제목 (배경 박스 포함)"""
    ax.add_patch(patches.Rectangle((x - 0.005, y - 0.018), 0.87, 0.026,
                                   color=LBLUE, transform=ax.transAxes,
                                   zorder=0))
    ax.text(x, y, title, fontproperties=KR_B, fontsize=size,
            color=BLUE, va='top', transform=ax.transAxes)
    return y - 0.032


def subsection(ax, x, y, title, size=9.5):
    ax.text(x, y, '>> ' + title, fontproperties=KR_B, fontsize=size,
            color=PURPLE, va='top', transform=ax.transAxes)
    return y - 0.024


def para(ax, x, y, lines, size=9, color=GRAY, indent=0):
    """여러 줄 한국어 텍스트"""
    if isinstance(lines, str):
        lines = [lines]
    cy = y
    for line in lines:
        ax.text(x + indent, cy, line, fontproperties=KR,
                fontsize=size, color=color, va='top',
                transform=ax.transAxes)
        cy -= 0.022
    return cy


def bullet(ax, x, y, items, size=9, color=GRAY):
    cy = y
    for item in items:
        ax.text(x, cy, '•', fontsize=size + 1, color=BLUE,
                va='top', transform=ax.transAxes)
        # 긴 줄 처리
        lines = textwrap.wrap(item, 80)
        for i, line in enumerate(lines):
            ax.text(x + 0.025, cy, line, fontproperties=KR,
                    fontsize=size, color=color, va='top',
                    transform=ax.transAxes)
            if i < len(lines) - 1:
                cy -= 0.020
        cy -= 0.022
    return cy


def code_block(ax, x, y, w, lines, title='Pseudocode', size=7.8):
    """코드 블록 박스 — 가용 공간에 맞게 자동으로 line spacing 조정."""
    n = len(lines)
    line_h = 0.019
    # 페이지 하단(y=0.02)을 넘지 않도록 line_h와 size를 축소
    avail = y - 0.02
    needed = line_h * (n + 1.5)
    if needed > avail > 0:
        line_h = avail / (n + 1.5)
        size   = max(5.5, size * (line_h / 0.019))
    h = line_h * (n + 1.5)
    # 배경
    ax.add_patch(patches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle='round,pad=0.005',
        facecolor='#263238', edgecolor='#37474F', linewidth=1,
        transform=ax.transAxes, zorder=1))
    # 제목 바
    ax.add_patch(patches.Rectangle((x, y - 0.022), w, 0.022,
                                   color='#37474F', transform=ax.transAxes, zorder=2))
    ax.text(x + 0.01, y - 0.006, title, fontproperties=MONO,
            fontsize=7, color='#80CBC4', va='top', transform=ax.transAxes, zorder=3)
    # 코드 라인
    colors = {'#': '#80CBC4', 'def': '#82B1FF', 'if': '#CF9FFF',
              'for': '#CF9FFF', 'while': '#CF9FFF', 'return': '#FF8A65',
              'INPUT': '#FFD54F', 'OUTPUT': '#A5D6A7', 'STEP': '#90CAF9'}
    cy = y - min(0.026, line_h + 0.007)
    for line in lines:
        stripped = line.lstrip()
        indent_n = len(line) - len(stripped)
        first_word = stripped.split()[0] if stripped.split() else ''
        c = '#ECEFF1'
        if stripped.startswith('#'):
            c = '#80CBC4'
        elif first_word in ('def', 'if', 'for', 'while', 'return'):
            c = '#CF9FFF'
        elif stripped.startswith('INPUT') or stripped.startswith('OUTPUT'):
            c = '#FFD54F'
        elif stripped.startswith('STEP') or stripped.startswith('// Step'):
            c = '#90CAF9'
        # 인라인 한국어 주석 분리: '#' 앞은 MONO, 뒤는 KR
        has_korean = any(ord(ch) > 0x3000 for ch in stripped)
        x_off = x + 0.012 + indent_n * 0.006
        # Fixed right column for Korean comments (avoids inaccurate width estimation)
        comment_col = x + w * 0.58
        if has_korean and '#' in stripped and not stripped.startswith('#'):
            comment_idx = stripped.index('#')
            code_part    = stripped[:comment_idx].rstrip()
            comment_part = stripped[comment_idx:]
            if code_part:
                ax.text(x_off, cy, code_part,
                        fontproperties=MONO, fontsize=size, color='#ECEFF1',
                        va='top', transform=ax.transAxes, zorder=3)
            ax.text(comment_col, cy, comment_part,
                    fontproperties=KR, fontsize=size - 0.5, color='#80CBC4',
                    va='top', transform=ax.transAxes, zorder=3)
        elif has_korean:
            # Pure Korean line (e.g., full-line Korean comment starting with #)
            fp_line = KR
            ax.text(x_off, cy, stripped,
                    fontproperties=fp_line, fontsize=size, color=c,
                    va='top', transform=ax.transAxes, zorder=3)
        else:
            ax.text(x_off, cy, stripped,
                    fontproperties=MONO, fontsize=size, color=c,
                    va='top', transform=ax.transAxes, zorder=3)
        cy -= line_h
    return y - h - 0.010


def eq_box(ax, x, y, w, latex_lines, label=None, size=10):
    """
    수식 박스 (배경 포함).
    - 순수 수식 라인 → mathtext (DejaVu Sans)
    - 한국어만 있는 라인 → KR 폰트
    - 수식 + 한국어 혼합 라인 → 수식을 왼쪽 중앙, 한국어를 오른쪽에 나란히 렌더링
      (줄 수를 늘리지 않아 박스 높이가 유지됨)
    """
    n = len(latex_lines)
    h = 0.032 * n + 0.020
    ax.add_patch(patches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle='round,pad=0.005',
        facecolor=LGREEN, edgecolor='#4CAF50', linewidth=0.8,
        transform=ax.transAxes))
    if label:
        ax.text(x + w - 0.01, y - 0.008, label,
                fontproperties=KR, fontsize=7, color='#388E3C',
                ha='right', va='top', transform=ax.transAxes)
    cy = y - 0.016
    for lat in latex_lines:
        has_korean = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in lat)
        if has_korean and '$' in lat:
            # 수식 부분과 한국어 부분을 동일 줄에 나란히 배치
            parts      = re.split(r'(\$[^$]+\$)', lat)
            math_parts = [p for p in parts if p.startswith('$') and p.endswith('$')]
            kr_parts   = [p.strip() for p in parts
                          if not (p.startswith('$') and p.endswith('$')) and p.strip()]
            math_str = '  '.join(math_parts)
            kr_str   = '  '.join(kr_parts)
            # 수식은 박스 왼쪽 60% 중앙, 한국어는 오른쪽 끝 정렬
            ax.text(x + w * 0.38, cy, math_str, fontsize=size, color='#1B5E20',
                    ha='center', va='top', transform=ax.transAxes)
            if kr_str:
                ax.text(x + w - 0.012, cy, kr_str, fontproperties=KR,
                        fontsize=size - 2.5, color='#388E3C',
                        ha='right', va='top', transform=ax.transAxes)
        elif has_korean:
            ax.text(x + w / 2, cy, lat, fontproperties=KR,
                    fontsize=size - 1, color='#1B5E20',
                    ha='center', va='top', transform=ax.transAxes)
        else:
            ax.text(x + w / 2, cy, lat, fontsize=size, color='#1B5E20',
                    ha='center', va='top', transform=ax.transAxes)
        cy -= 0.032
    return y - h - 0.012


def info_box(ax, x, y, w, lines, color=LBLUE, border='#1565C0', size=8.8):
    n = sum(len(textwrap.wrap(l, 90)) for l in lines) if lines else 1
    h = 0.022 * (n + 0.5) + 0.010
    ax.add_patch(patches.FancyBboxPatch(
        (x, y - h), w, h, boxstyle='round,pad=0.005',
        facecolor=color, edgecolor=border, linewidth=0.8,
        transform=ax.transAxes))
    cy = y - 0.016
    for line in lines:
        wrapped = textwrap.wrap(line, 88)
        for wl in wrapped:
            ax.text(x + 0.015, cy, wl, fontproperties=KR,
                    fontsize=size, color=GRAY, va='top',
                    transform=ax.transAxes)
            cy -= 0.021
    return y - h - 0.010


def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # 배경
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='#0D1B2A'))

    # 상단 장식 바
    for i, (c, y_) in enumerate(zip(
            ['#1565C0','#1976D2','#1E88E5','#2196F3'],
            [0.97, 0.96, 0.95, 0.94])):
        ax.add_patch(patches.Rectangle((0, y_), 1, 0.01, color=c, alpha=0.8))

    # 중앙 타이틀
    ax.text(0.5, 0.76, 'Visual-Inertial Odometry',
            fontsize=28, color='white', ha='center', va='center',
            fontweight='bold', fontfamily='DejaVu Sans')
    ax.text(0.5, 0.69, '이론과 구현 — 학습 안내서',
            fontsize=20, color='#90CAF9', ha='center', va='center',
            fontproperties=KR_B)

    # 부제
    ax.text(0.5, 0.62,
            'Intel RealSense T265  ·  Extended Kalman Filter  ·  Stereo VO',
            fontsize=11, color='#78909C', ha='center', va='center',
            fontfamily='DejaVu Sans')

    # 구분선
    ax.add_patch(patches.Rectangle((0.15, 0.595), 0.70, 0.002, color='#1565C0'))

    # 챕터 목록
    chapters = [
        ('Ch 1', 'VIO 개요 — 왜 필요한가?'),
        ('Ch 2', '3D 공간과 쿼터니언'),
        ('Ch 3', 'IMU 센서 모델'),
        ('Ch 4', 'Extended Kalman Filter 이론'),
        ('Ch 5', 'IMU 예측 단계 — F 행렬과 Q 행렬'),
        ('Ch 6', '중력 정렬 초기화'),
        ('Ch 7', '피시아이 카메라 — Kannala-Brandt'),
        ('Ch 8', '스테레오 비주얼 오도메트리'),
        ('Ch 9', 'EKF 측정 업데이트와 게이팅'),
        ('Ch 10', 'ZUPT — 정지 검출'),
        ('Ch 11', '전체 시스템 파이프라인'),
    ]
    cy = 0.565
    for num, title in chapters:
        ax.text(0.28, cy, num, fontsize=8.5, color='#2196F3',
                ha='right', va='top', fontfamily='DejaVu Sans', fontweight='bold')
        ax.text(0.30, cy, title, fontproperties=KR,
                fontsize=8.5, color='#CFD8DC', ha='left', va='top')
        cy -= 0.040

    # 하단
    ax.text(0.5, 0.055,
            'Intel RealSense T265  |  Python / OpenCV / NumPy',
            fontsize=8, color='#455A64', ha='center', va='center',
            fontfamily='DejaVu Sans')
    ax.add_patch(patches.Rectangle((0, 0.04), 1, 0.002, color='#1565C0'))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch1(pdf):
    fig, ax = new_page(pdf, title='Chapter 1  —  VIO란 무엇인가?', chapter='Ch 1 · 개요')
    y = 0.895

    y = section(ax, MARGIN_L, y, '1.1  자기 위치 추정 (Ego-motion Estimation)')
    y = para(ax, MARGIN_L, y, [
        '로봇이나 드론이 자신의 위치와 자세를 실시간으로 추정하는 문제를 자기 위치 추정 (Ego-motion',
        'Estimation) 이라고 합니다. GPS가 없는 실내 환경이나 빠른 움직임이 필요한 상황에서는',
        '카메라와 IMU(관성 측정 장치)를 결합한 VIO(Visual-Inertial Odometry)를 사용합니다.',
    ])
    y -= 0.010

    y = section(ax, MARGIN_L, y, '1.2  센서별 특성과 상보적 관계')
    # 센서 비교 박스
    box_data = [
        ('IMU (가속도계 + 자이로)', '#E3F2FD', '#1565C0',
         ['[+] 200 Hz 고주파 측정', '[+] 빠른 움직임에 강함',
          '[-] 이중 적분 시 오차 누적 (드리프트)', '[-] 스케일 절대값 불명확']),
        ('카메라 (스테레오)', '#E8F5E9', '#2E7D32',
         ['[+] 30 Hz, 풍부한 텍스처 정보', '[+] 스테레오 베이스라인으로 metric 스케일',
          '[-] 빠른 움직임에 모션 블러', '[-] 어두운 환경에 취약']),
        ('VIO (IMU + 카메라)', '#FFF3E0', '#E65100',
         ['[+] IMU로 고주파 예측 (Predict)', '[+] 카메라로 오차 보정 (Update)',
          '[+] Metric 스케일 + 낮은 드리프트', '<-- 두 센서의 장점만 취함']),
    ]
    bx = MARGIN_L
    for title_b, bg, border, items in box_data:
        bw = 0.27
        h  = 0.115
        ax.add_patch(patches.FancyBboxPatch(
            (bx, y - h), bw, h, boxstyle='round,pad=0.005',
            facecolor=bg, edgecolor=border, linewidth=1.2,
            transform=ax.transAxes))
        ax.text(bx + bw/2, y - 0.010, title_b,
                fontproperties=KR_B, fontsize=8.5, color=border,
                ha='center', va='top', transform=ax.transAxes)
        iy = y - 0.032
        for item in items:
            ax.text(bx + 0.012, iy, item, fontproperties=KR,
                    fontsize=7.8, color=GRAY, va='top', transform=ax.transAxes)
            iy -= 0.018
        bx += bw + 0.025
    y -= 0.130

    y = section(ax, MARGIN_L, y, '1.3  Intel RealSense T265 하드웨어')
    # T265 스펙 박스
    specs = [
        ('카메라', '피시아이 스테레오 (FOV ~160°), 848×800, 30 fps'),
        ('IMU',    '가속도계 62 Hz, 자이로스코프 200 Hz, factory calibrated'),
        ('베이스라인', '6.4 cm (스테레오 baseline) — metric 깊이 복원의 기준'),
        ('렌즈 모델', 'Kannala-Brandt4 (피시아이 왜곡 4계수)'),
    ]
    ax.add_patch(patches.FancyBboxPatch(
        (MARGIN_L, y - 0.095), 0.87, 0.095, boxstyle='round,pad=0.005',
        facecolor='#F3E5F5', edgecolor=PURPLE, linewidth=1,
        transform=ax.transAxes))
    sy = y - 0.016
    for key, val in specs:
        ax.text(MARGIN_L + 0.012, sy, key + ':',
                fontproperties=KR_B, fontsize=8.5, color=PURPLE,
                va='top', transform=ax.transAxes)
        ax.text(MARGIN_L + 0.095, sy, val,
                fontproperties=KR, fontsize=8.5, color=GRAY,
                va='top', transform=ax.transAxes)
        sy -= 0.022
    y -= 0.108

    y = section(ax, MARGIN_L, y, '1.4  전체 파이프라인 한눈에 보기')

    # 파이프라인 다이어그램
    steps = [
        ('IMU\n데이터 수집', '#1565C0', 'white'),
        ('EKF\n예측 (Predict)', '#1976D2', 'white'),
        ('스테레오\n특징점 검출', '#2E7D32', 'white'),
        ('삼각측량\n(Depth)', '#388E3C', 'white'),
        ('PnP\n(Pose)', '#E65100', 'white'),
        ('EKF\n업데이트', '#C62828', 'white'),
        ('위치/자세\n출력', '#4A148C', 'white'),
    ]
    bx_ = MARGIN_L + 0.02
    bw_ = 0.105
    by_ = y - 0.035
    bh_ = 0.055
    for i, (label, bg, fg) in enumerate(steps):
        ax.add_patch(patches.FancyBboxPatch(
            (bx_, by_), bw_, bh_, boxstyle='round,pad=0.004',
            facecolor=bg, edgecolor='white', linewidth=0.5,
            transform=ax.transAxes))
        ax.text(bx_ + bw_/2, by_ + bh_/2, label,
                fontproperties=KR, fontsize=7.2, color=fg,
                ha='center', va='center', transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate('', xy=(bx_ + bw_ + 0.012, by_ + bh_/2),
                        xytext=(bx_ + bw_, by_ + bh_/2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2))
        bx_ += bw_ + 0.015
    y -= 0.095

    # 200 Hz / 30 Hz 레이블
    ax.text(MARGIN_L + 0.065, y + 0.005, '←── 200 Hz ──→',
            fontsize=7.5, color='#1565C0', ha='center',
            transform=ax.transAxes, style='italic')
    ax.text(MARGIN_L + 0.38, y + 0.005, '←──────── 30 Hz ─────────→',
            fontsize=7.5, color='#2E7D32', ha='center',
            transform=ax.transAxes, style='italic')

    y -= 0.020
    y = info_box(ax, MARGIN_L, y, 0.87, [
        '핵심 개념:  IMU는 빠르지만 드리프트, 카메라는 느리지만 정확합니다.',
        'EKF가 두 센서를 통계적으로 최적 융합하여 고주파 + 낮은 드리프트를 동시에 달성합니다.',
    ], color='#FFF9C4', border='#F9A825')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch2(pdf):
    fig, ax = new_page(pdf, title='Chapter 2  —  3D 공간과 쿼터니언', chapter='Ch 2 · 수학 기초')
    y = 0.895

    y = section(ax, MARGIN_L, y, '2.1  좌표계 (Reference Frames)')
    y = para(ax, MARGIN_L, y, [
        'VIO에서는 세 가지 좌표계가 사용됩니다:',
    ])
    items_frame = [
        '세계 좌표계 (World frame, W):  초기화 시점에 고정된 관성 기준계. Z축이 중력 반대 방향 (+위쪽).',
        '몸체 좌표계 (Body frame, B):  IMU가 붙어 있는 디바이스. 디바이스와 함께 회전/이동.',
        '카메라 좌표계 (Camera frame, C):  왼쪽 카메라의 광학 중심. Z축이 카메라가 바라보는 방향.',
    ]
    y = bullet(ax, MARGIN_L, y, items_frame)
    y -= 0.005

    y = section(ax, MARGIN_L, y, '2.2  회전 행렬 (Rotation Matrix)')
    y = para(ax, MARGIN_L, y, [
        '3×3 직교 행렬 R로 회전을 표현합니다. R의 열벡터는 회전 후 좌표계의 축 방향입니다.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\mathbf{R} \in SO(3),\quad \mathbf{R}^\top\mathbf{R} = \mathbf{I},\quad \det(\mathbf{R}) = 1$',
                r'$\mathbf{v}_W = \mathbf{R}_{BW}\,\mathbf{v}_B$   (몸체 → 세계 변환)'],
               label='성질')
    y -= 0.005

    y = section(ax, MARGIN_L, y, '2.3  쿼터니언 (Quaternion) — 왜 쓰는가?')
    y = para(ax, MARGIN_L, y, [
        '회전 행렬은 9개 숫자지만 자유도는 3입니다. 쿼터니언은 4개 숫자로 모든 회전을 표현하며',
        '짐벌 락(Gimbal Lock)이 없고 수치 오차가 누적되어도 재정규화가 쉽습니다.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\mathbf{q} = [w,\; x,\; y,\; z]^\top,\quad w^2+x^2+y^2+z^2 = 1$',
                r'해석: 축 $\hat{n}$으로 각도 $\theta$ 회전 $\Rightarrow$  $\mathbf{q} = [\cos\frac{\theta}{2},\; \hat{n}\sin\frac{\theta}{2}]$'],
               label='정의')

    y = subsection(ax, MARGIN_L, y, '쿼터니언 곱셈 (Hamilton Product) — 두 회전의 합성')
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$w = w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2$',
                r'$x = w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2$',
                r'$y = w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2$',
                r'$z = w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2$'],
               label='Hamilton Product', size=9)

    y = subsection(ax, MARGIN_L, y, '쿼터니언 → 회전 행렬')
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$R_{00}=1-2(y^2+z^2),\; R_{01}=2(xy-wz),\; R_{02}=2(xz+wy)$',
                r'$R_{10}=2(xy+wz),\; R_{11}=1-2(x^2+z^2),\; R_{12}=2(yz-wx)$',
                r'$R_{20}=2(xz-wy),\; R_{21}=2(yz+wx),\; R_{22}=1-2(x^2+y^2)$'],
               label='R(q)', size=9.5)

    y -= 0.005
    y = info_box(ax, MARGIN_L, y, 0.87, [
        '직관:  q = [1, 0, 0, 0] 은 단위 회전(항등)입니다. q_w = cos(θ/2)이므로 작은 각도에서 w ≈ 1.',
        '정규화:  통합 오차로 ||q|| ≠ 1이 되면 q / ||q|| 로 재정규화합니다 (매 EKF 스텝마다 수행).',
    ], color='#FFF9C4', border='#F9A825')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch3(pdf):
    fig, ax = new_page(pdf, title='Chapter 3  —  IMU 센서 모델', chapter='Ch 3 · 센서 모델')
    y = 0.895

    y = section(ax, MARGIN_L, y, '3.1  가속도계 모델')
    y = para(ax, MARGIN_L, y, [
        '가속도계는 비력(Specific Force) = 실제 가속도 − 중력 을 몸체 좌표계에서 측정합니다.',
        '측정값에는 바이어스와 가우시안 노이즈가 섞입니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\tilde{\mathbf{a}} = \mathbf{R}^\top(\mathbf{a}_W - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$',
                r'$\mathbf{n}_a \sim \mathcal{N}(\mathbf{0},\;\sigma_a^2\mathbf{I}),\quad '
                r'\dot{\mathbf{b}}_a \sim \mathcal{N}(\mathbf{0},\;\sigma_{ba}^2\mathbf{I})$'],
               label='가속도계')
    y = para(ax, MARGIN_L, y, [
        '여기서 R은 세계→몸체 회전, g = [0, 0, −9.81] m/s² (세계 좌표 중력), n_a는 백색 노이즈,',
        'b_a는 천천히 변하는 바이어스(랜덤 워크)입니다. T265 스펙: σ_a ≈ 0.01 m/s²/√Hz.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '3.2  자이로스코프 모델')
    y = para(ax, MARGIN_L, y, [
        '자이로는 각속도 ω를 몸체 좌표계에서 측정합니다 (단위: rad/s):',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\tilde{\mathbf{\omega}} = \mathbf{\omega} + \mathbf{b}_g + \mathbf{n}_g$',
                r'$\mathbf{n}_g \sim \mathcal{N}(\mathbf{0},\;\sigma_g^2\mathbf{I}),\quad '
                r'\dot{\mathbf{b}}_g \sim \mathcal{N}(\mathbf{0},\;\sigma_{bg}^2\mathbf{I})$'],
               label='자이로스코프')
    y = para(ax, MARGIN_L, y, [
        'T265 스펙: σ_g ≈ 0.002 rad/s/√Hz. 자이로 바이어스는 정지 상태에서 측정값 평균으로',
        '초기 추정합니다 (Ch 6 참조).',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '3.3  바이어스의 의미와 중요성')
    # 바이어스 다이어그램
    bx_ = MARGIN_L + 0.02
    ax.add_patch(patches.FancyBboxPatch(
        (bx_, y - 0.120), 0.83, 0.115, boxstyle='round,pad=0.005',
        facecolor='#F3E5F5', edgecolor=PURPLE, linewidth=0.8,
        transform=ax.transAxes))
    ax.text(bx_ + 0.01, y - 0.012,
            '왜 바이어스 추정이 중요한가?',
            fontproperties=KR_B, fontsize=9.5, color=PURPLE,
            va='top', transform=ax.transAxes)
    bias_text = [
        '• 가속도계 바이어스 b_a가 0.01 m/s² 오차가 있으면:',
        '    속도 오차 = 0.01 × t   →   10초 후 0.1 m/s 속도 오차',
        '    위치 오차 = 0.5 × 0.01 × t²  →   10초 후 0.5 m 위치 오차',
        '• 자이로 바이어스 b_g가 0.01 rad/s 오차가 있으면:',
        '    각도 오차 = 0.01 × t   →   10초 후 0.1 rad ≈ 5.7° 자세 오차',
        '• EKF가 이 바이어스를 상태로 추정하여 실시간 보정합니다.',
    ]
    by_inner = y - 0.036
    for bt in bias_text:
        ax.text(bx_ + 0.012, by_inner, bt, fontproperties=KR,
                fontsize=8.5, color=GRAY, va='top', transform=ax.transAxes)
        by_inner -= 0.018
    y -= 0.130

    y = section(ax, MARGIN_L, y, '3.4  IMU 전처리: 바이어스 보정과 세계 좌표 변환')
    y = para(ax, MARGIN_L, y, [
        'EKF 예측 단계에서 IMU 측정값을 사용하기 전에 추정된 바이어스를 제거합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\mathbf{a}_{body} = \tilde{\mathbf{a}} - \hat{\mathbf{b}}_a$   (바이어스 보정 가속도)',
                r'$\mathbf{\omega}_{body} = \tilde{\mathbf{\omega}} - \hat{\mathbf{b}}_g$   (바이어스 보정 각속도)',
                r'$\mathbf{a}_W = \mathbf{R}\,\mathbf{a}_{body} + \mathbf{g}$   (세계 좌표 가속도)'],
               label='IMU 전처리')

    y = info_box(ax, MARGIN_L, y, 0.87, [
        '직관:  IMU는 항상 몸체 좌표계에서 값을 줍니다. EKF의 상태는 세계 좌표계 기준이므로',
        '       회전 행렬 R로 변환해야 합니다. 이 변환이 틀리면 중력 방향이 뒤섞여 발산합니다.',
    ], color='#FFF9C4', border='#F9A825')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch4(pdf):
    fig, ax = new_page(pdf, title='Chapter 4  —  Extended Kalman Filter', chapter='Ch 4 · EKF 이론')
    y = 0.895

    y = section(ax, MARGIN_L, y, '4.1  칼만 필터의 직관 — 두 정보의 최적 융합')
    y = para(ax, MARGIN_L, y, [
        '당신이 위치를 두 가지 방법으로 추정한다고 가정합니다:',
        '  ① 지도로 예측: "10m 앞에 있을 것" (불확실도 ±2 m)',
        '  ② GPS로 측정: "현재 위치는 11m" (불확실도 ±1 m)',
        '최적 답은 두 정보를 불확실도에 반비례하여 가중 평균한 것입니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\hat{x} = \frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2}\,x_1 + \frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}\,x_2$',
                r'칼만 게인 $K = \frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}$ 는 예측을 얼마나 믿을지 결정'],
               label='1D 직관')
    y -= 0.005

    y = section(ax, MARGIN_L, y, '4.2  EKF 상태 벡터 x (16차원)')
    ax.add_patch(patches.FancyBboxPatch(
        (MARGIN_L + 0.02, y - 0.112), 0.83, 0.108, boxstyle='round,pad=0.005',
        facecolor='#E3F2FD', edgecolor=BLUE, linewidth=1,
        transform=ax.transAxes))
    state_items = [
        ('[0:3]   위치  p = (p_x, p_y, p_z)',           '단위: m,   세계 좌표계'),
        ('[3:6]   속도  v = (v_x, v_y, v_z)',           '단위: m/s, 세계 좌표계'),
        ('[6:10]  쿼터니언  q = (q_w, q_x, q_y, q_z)', '||q|| = 1, 세계←몸체'),
        ('[10:13] 가속도 바이어스  b_a',                'EKF가 실시간 추정'),
        ('[13:16] 자이로 바이어스  b_g',                'EKF가 실시간 추정'),
    ]
    sy_ = y - 0.018
    for state, note in state_items:
        has_kr_s = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in state)
        ax.text(MARGIN_L + 0.04, sy_, state,
                fontproperties=KR if has_kr_s else MONO,
                fontsize=8, color='#0D47A1',
                va='top', transform=ax.transAxes)
        ax.text(MARGIN_L + 0.56, sy_, note,
                fontproperties=KR, fontsize=7.8, color='#1565C0',
                va='top', transform=ax.transAxes)
        sy_ -= 0.021
    y -= 0.120

    y = section(ax, MARGIN_L, y, '4.3  공분산 행렬 P (16×16) — 상태 불확실도')
    y = para(ax, MARGIN_L, y, [
        'P는 상태 벡터 x의 각 성분 간 공분산을 담은 행렬입니다.',
        'P[i,i]가 크다 = 해당 상태 성분이 불확실하다 = Kalman 게인이 커진다 = 측정을 더 신뢰한다.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\mathbf{P} = E[(\mathbf{x}-\hat{\mathbf{x}})(\mathbf{x}-\hat{\mathbf{x}})^\top]$',
                r'초기값: $P_{pos}=0.005,\; P_{vel}=0.05,\; P_{att}=10^{-4}$ (단위: m², m²/s², rad²)'],
               label='공분산')
    y -= 0.005

    y = section(ax, MARGIN_L, y, '4.4  EKF 2단계 구조')
    # 예측/업데이트 박스
    for i, (title_b, desc, bg, border) in enumerate([
        ('PREDICT  (예측)',
         '이전 상태 + IMU 입력 → 새 상태 예측\n'
         'x_{k+1|k} = f(x_k, u_k),   P = F P F^T + Q',
         '#E3F2FD', '#1565C0'),
        ('UPDATE  (업데이트)',
         '카메라 측정으로 예측 오차 보정\n'
         'x_{k|k} = x_{k+1|k} + K(z - Hx),   P = (I-KH)P(I-KH)^T + KRK^T',
         '#E8F5E9', '#2E7D32'),
    ]):
        bx_i = MARGIN_L + i * 0.45
        ax.add_patch(patches.FancyBboxPatch(
            (bx_i, y - 0.090), 0.42, 0.090, boxstyle='round,pad=0.005',
            facecolor=bg, edgecolor=border, linewidth=1.2,
            transform=ax.transAxes))
        ax.text(bx_i + 0.21, y - 0.012, title_b,
                fontproperties=KR_B, fontsize=9.5, color=border,
                ha='center', va='top', transform=ax.transAxes)
        dy_ = y - 0.035
        for dline in desc.split('\n'):
            has_kr = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in dline)
            ax.text(bx_i + 0.210, dy_, dline,
                    fontproperties=KR if has_kr else MONO,
                    fontsize=7.2, color=GRAY,
                    ha='center', va='top', transform=ax.transAxes)
            dy_ -= 0.018
    # 화살표: 예측→업데이트, 업데이트→예측
    arrow(ax, MARGIN_L + 0.42, y - 0.045, MARGIN_L + 0.45, y - 0.045)
    ax.text(MARGIN_L + 0.435, y - 0.030, '측정\n도착',
            fontproperties=KR, fontsize=6.5, color=ORANGE,
            ha='center', va='top', transform=ax.transAxes)
    y -= 0.100

    y = info_box(ax, MARGIN_L, y, 0.87, [
        'EKF와 KF의 차이:  IMU 적분, 쿼터니언 갱신, 회전 변환 등은 모두 비선형 함수입니다.',
        'EKF는 이 비선형 함수를 현재 상태 주변에서 1차 Taylor 전개(Jacobian)로 선형화하여 처리합니다.',
    ], color='#FFF9C4', border='#F9A825')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch5(pdf):
    fig, ax = new_page(pdf, title='Chapter 5  —  IMU 예측 단계', chapter='Ch 5 · Prediction Step')
    y = 0.895

    y = section(ax, MARGIN_L, y, '5.1  운동 방정식 (Equations of Motion)')
    y = para(ax, MARGIN_L, y, [
        'IMU 입력으로 이전 상태에서 다음 상태를 예측합니다. dt = 1/200 s (200 Hz 자이로 기준).',
    ])
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k\,dt + \frac{1}{2}\,\mathbf{a}_W\,dt^2$',
                r'$\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_W\,dt,\quad'
                r'\mathbf{a}_W = \mathbf{R}(\mathbf{q}_k)\,(\tilde{\mathbf{a}}-\mathbf{b}_a) + \mathbf{g}$',
                r'$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \exp(tfrac{1}{2}\,\mathbf{\omega}_{body}\,dt)$'],
               label='이산 운동 방정식', size=9.5)
    y = para(ax, MARGIN_L, y, [
        '쿼터니언 적분 상세: 각속도 벡터 ω의 크기가 angle = ||ω|| × dt 일 때:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$d\mathbf{q} = [\cos\frac{angle}{2},\; \frac{\mathbf{\omega}}{|\mathbf{\omega}|}\sin\frac{angle}{2}]$   (각도 크면)',
                r'$d\mathbf{q} \approx [1,\; \frac{1}{2}\mathbf{\omega}\,dt]$   (각도 매우 작으면, 1차 근사)'],
               label='쿼터니언 적분')
    y -= 0.005

    y = section(ax, MARGIN_L, y, '5.2  선형화 — Jacobian F 행렬 (16×16)')
    y = para(ax, MARGIN_L, y, [
        'EKF 예측 시 공분산 전파를 위해 운동 방정식의 Jacobian F = ∂f/∂x 가 필요합니다.',
        '대부분의 F는 단위 행렬이고, 다음 블록만 비단위 값을 가집니다:',
    ])
    # F 행렬 구조 시각화
    ax.add_patch(patches.FancyBboxPatch(
        (MARGIN_L + 0.02, y - 0.130), 0.83, 0.126, boxstyle='round,pad=0.005',
        facecolor=LGRAY, edgecolor='#9E9E9E', linewidth=0.8,
        transform=ax.transAxes))
    f_items = [
        ('F[p, v]   = I × dt',             '위치는 속도 × dt만큼 변함'),
        ('F[p, ba]  = -½R × dt²',          '바이어스 오차가 위치에 미치는 영향'),
        ('F[p, q]   = ½ ∂(Ra)/∂q × dt²', '자세 오차가 가속도 방향 변환에 미치는 영향'),
        ('F[v, q]   = ∂(Ra)/∂q × dt',     '자세 오차가 속도 변화에 미치는 영향'),
        ('F[v, ba]  = -R × dt',            '바이어스 오차가 속도에 미치는 영향'),
        ('F[q, bg]  = ∂q_new/∂bg × dt',   '자이로 바이어스 오차가 자세에 미치는 영향'),
    ]
    fy_ = y - 0.018
    for formula, note in f_items:
        ax.text(MARGIN_L + 0.04, fy_, formula,
                fontproperties=MONO, fontsize=8, color='#0D47A1',
                va='top', transform=ax.transAxes)
        ax.text(MARGIN_L + 0.52, fy_, note,
                fontproperties=KR, fontsize=7.8, color=GRAY,
                va='top', transform=ax.transAxes)
        fy_ -= 0.021
    y -= 0.140

    y = section(ax, MARGIN_L, y, '5.3  프로세스 노이즈 Q — 모델 불확실도')
    y = para(ax, MARGIN_L, y, [
        'Q는 IMU 노이즈가 상태에 미치는 불확실도를 표현합니다. σ_a, σ_g 가 클수록 Q가 커지고',
        'P가 빠르게 성장하여 다음 측정에서의 Kalman 게인이 커집니다.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\mathbf{Q}[v,v] = \sigma_a^2 \cdot dt \cdot \mathbf{I}_3$',
                r'$\mathbf{Q}[p,p] = \frac{1}{4}\sigma_a^2 \cdot dt^3 \cdot \mathbf{I}_3$',
                r'$\mathbf{Q}[q,q] = \frac{1}{4}\sigma_g^2 \cdot dt \cdot \mathbf{I}_4$',
                r'$\mathbf{P} \leftarrow \mathbf{F}\mathbf{P}\mathbf{F}^\top + \mathbf{Q}$'],
               label='Q 구성 및 공분산 전파', size=9.5)

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# IMU 예측 스텝 의사코드',
        'def predict(x, P, accel_raw, gyro_raw, dt):',
        '    p, v, q, ba, bg = unpack(x)',
        '    R = quat_to_rot(q)',
        '    a_body = accel_raw - ba              # 바이어스 보정',
        '    w_body = gyro_raw  - bg',
        '    a_world = R @ a_body + [0, 0, -9.81] # 세계 좌표 가속도',
        '',
        '    p_new = p + v*dt + 0.5*a_world*dt**2  # 위치 적분',
        '    v_new = v + a_world*dt                 # 속도 적분',
        '    dq = angle_to_quat(w_body * dt)        # 쿼터니언 적분',
        '    q_new = normalize(quat_mult(q, dq))',
        '',
        '    F = build_jacobian(a_body, q, dt)      # 선형화',
        '    Q = build_process_noise(sigma_a, sigma_g, dt)',
        '    P_new = F @ P @ F.T + Q                # 공분산 전파',
        '    return pack(p_new, v_new, q_new, ba, bg), P_new',
    ], title='Pseudocode — EKF Predict')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch6(pdf):
    fig, ax = new_page(pdf, title='Chapter 6  —  중력 정렬 초기화', chapter='Ch 6 · 초기화')
    y = 0.895

    y = section(ax, MARGIN_L, y, '6.1  왜 초기화가 필요한가?')
    y = para(ax, MARGIN_L, y, [
        'EKF의 초기 쿼터니언은 q = [1,0,0,0] (항등 회전)으로 설정되어 있습니다.',
        '이는 몸체 좌표계 = 세계 좌표계로 가정하는 것입니다. 하지만 실제 디바이스가',
        '기울어져 있다면, IMU로부터 변환한 중력 방향이 틀려서 EKF가 즉시 발산합니다.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '6.2  정지 상태에서의 가속도 = 중력 벡터')
    y = para(ax, MARGIN_L, y, [
        '가속도계 모델에서 디바이스가 정지하면 a_world = 0이므로:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\tilde{\mathbf{a}} \approx \mathbf{R}^\top(0 - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$',
                r'$\approx -\mathbf{R}^\top\mathbf{g}$   (바이어스 · 노이즈 무시)',
                r'즉, 정지 시 가속도계 측정값은 중력 벡터가 몸체 좌표로 표현된 것'],
               label='정지 가속도 = 중력')
    y = para(ax, MARGIN_L, y, [
        '따라서 1초간의 정지 가속도 평균 g_body = mean(ã)는 몸체 좌표계의 중력 방향입니다.',
        '세계 좌표계에서 중력은 z_world = [0,0,1] (−g 방향 기준)로 알려져 있으므로,',
        '두 벡터를 정렬하는 회전을 초기 쿼터니언으로 사용합니다.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '6.3  축-각도 방법으로 초기 쿼터니언 계산')
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$\hat{\mathbf{g}}_{body} = \frac{\mathbf{g}_{body}}{||\mathbf{g}_{body}||}$',
                r'$\mathbf{axis} = \hat{\mathbf{g}}_{body} \times \mathbf{z}_{world},\quad '
                r'\theta = \arctan2(||\mathbf{axis}||,\; \hat{\mathbf{g}}\cdot\mathbf{z})$',
                r'$\mathbf{q}_{init} = [\cos\frac{\theta}{2},\; \hat{\mathbf{axis}}\sin\frac{\theta}{2}]$'],
               label='중력 정렬 쿼터니언', size=9.5)

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# 중력 정렬 초기화 의사코드',
        'def initialize_from_accel(accel_samples, gyro_samples):',
        '    # 1초간(~60 샘플) 정지 데이터 평균',
        '    g_body = mean(accel_samples)        # 몸체 좌표 중력 벡터',
        '    g_hat  = g_body / norm(g_body)      # 단위 벡터',
        '    z_world = [0, 0, 1]                 # 세계 좌표 중력 방향 (위쪽)',
        '',
        '    # 두 벡터 사이의 회전 = 초기 자세',
        '    axis  = cross(g_hat, z_world)       # 회전축',
        '    sin_a = norm(axis)',
        '    if sin_a < 1e-6:                    # 이미 정렬됨 or 완전 반대',
        '        q = [1,0,0,0] if g_hat[2] > 0 else [0,1,0,0]',
        '    else:',
        '        axis /= sin_a',
        '        angle = atan2(sin_a, dot(g_hat, z_world))',
        '        q = [cos(angle/2), axis * sin(angle/2)]',
        '',
        '    x[6:10] = normalize(q)              # EKF 초기 쿼터니언 설정',
        '    x[13:16] = mean(gyro_samples)       # 자이로 바이어스 초기 추정',
        '    imu_buffer.reset()                  # 정렬 전 IMU 샘플 버림',
    ], title='Pseudocode — Gravity Alignment')

    y = info_box(ax, MARGIN_L, y, 0.87, [
        '핵심:  중력 정렬 없이 EKF를 시작하면 a_world = R @ a_body + g 에서 R이 잘못되어',
        '       중력이 소거되지 않고 가속도 신호에 9.81 m/s² 오프셋이 남아 위치가 수백 cm/s² 발산합니다.',
    ], color='#FFEBEE', border='#C62828')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch7(pdf):
    fig, ax = new_page(pdf, title='Chapter 7  —  피시아이 카메라 모델', chapter='Ch 7 · 카메라 모델')
    y = 0.895

    y = section(ax, MARGIN_L, y, '7.1  핀홀 카메라 모델 (표준 카메라)')
    y = para(ax, MARGIN_L, y, [
        '일반 카메라는 3D 점을 2D 이미지로 투영할 때 원근 투영(Perspective Projection)을 사용합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$u = f_x \cdot \frac{X}{Z} + c_x,\quad v = f_y \cdot \frac{Y}{Z} + c_y$',
                r'$\mathbf{K}=diag(f_x, f_y, 1)$ with principal point $(c_x, c_y)$'],
               label='핀홀 투영', size=10)
    y = para(ax, MARGIN_L, y, [
        'f_x, f_y: 초점거리(픽셀 단위),  c_x, c_y: 주점(이미지 중심).  T265: f ≈ 287 px.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '7.2  피시아이 카메라 — Kannala-Brandt 모델')
    y = para(ax, MARGIN_L, y, [
        'T265는 약 160° FOV의 피시아이 렌즈를 사용합니다. 광각에서 핀홀 모델은 맞지 않으며',
        'Kannala-Brandt 4계수 모델을 사용합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$\theta = \arctan\!\left(\frac{\sqrt{X^2+Y^2}}{Z}\right)$   (입사각)',
                r'$r_d = \theta + k_1\theta^3 + k_2\theta^5 + k_3\theta^7 + k_4\theta^9$   (방사 왜곡)',
                r'$u = f_x\,\frac{X}{\sqrt{X^2+Y^2}}\,r_d + c_x$'],
               label='Kannala-Brandt', size=9.5)
    y = para(ax, MARGIN_L, y, [
        '핀홀은 tan(θ)를 사용하지만 KB는 θ + 다항식 보정을 사용하여 광각에서도 정확합니다.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '7.3  왜곡 보정 (Undistortion)')
    y = para(ax, MARGIN_L, y, [
        '피시아이 이미지 → 핀홀 이미지로 변환합니다. 변환 후 이미지는 왜곡 없이 표준',
        '카메라 모델이 적용됩니다. 출력 카메라 행렬 K_new를 어떻게 설정하느냐가 중요합니다:',
    ])

    # K_new 비교 박스
    for i, (label, bg, border, desc, outcome) in enumerate([
        ('K_new = K1 (원래 초점거리 유지)',
         '#E8F5E9', '#2E7D32',
         '원래 카메라와 같은 FOV / 크기',
         '[*] 사용 (자연스러운 모습)'),
        ('K_new = estimateNewCameraMatrix\n(balance=0.3)',
         '#FFEBEE', '#C62828',
         'f_new ≈ 112 px (원래의 39%)',
         '[*] 2.5× 줌아웃 — 이미지가 "펴짐"'),
    ]):
        bx_i = MARGIN_L + i * 0.45
        ax.add_patch(patches.FancyBboxPatch(
            (bx_i, y - 0.090), 0.42, 0.090, boxstyle='round,pad=0.005',
            facecolor=bg, edgecolor=border, linewidth=1.2,
            transform=ax.transAxes))
        ax.text(bx_i + 0.210, y - 0.012, label,
                fontproperties=KR_B, fontsize=8.5, color=border,
                ha='center', va='top', transform=ax.transAxes)
        ax.text(bx_i + 0.210, y - 0.040, desc,
                fontproperties=KR, fontsize=8, color=GRAY,
                ha='center', va='top', transform=ax.transAxes)
        ax.text(bx_i + 0.210, y - 0.068, outcome,
                fontproperties=KR_B, fontsize=8, color=border,
                ha='center', va='top', transform=ax.transAxes)
    y -= 0.100

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# 피시아이 → 핀홀 왜곡 보정',
        'def undistort(img, K, D, K_new):',
        '    # cv2.fisheye.undistortImage:',
        '    #   K     = 원래 내부 파라미터 (Kannala-Brandt)',
        '    #   D     = 왜곡 계수 [k1, k2, k3, k4]',
        '    #   K_new = 출력 핀홀 카메라 행렬',
        '    # → 각 출력 픽셀 (u,v)에 대해 역 KB 모델로 입력 픽셀 좌표를',
        '    #   계산하여 bilinear interpolation 으로 샘플링',
        '    h, w = img.shape[:2]',
        '    return cv2.fisheye.undistortImage(img, K, D, None, K_new, (w, h))',
    ], title='Pseudocode — Fisheye Undistortion')

    y = info_box(ax, MARGIN_L, y, 0.87, [
        '핵심 버그 사례:  balance=0.3으로 자동 계산된 K_new를 사용하면 f_new << f_original이 되어',
        '이미지가 심하게 "펴져" 보입니다. K_new = K1.copy()로 원래 초점거리를 유지해야 합니다.',
    ], color='#FFEBEE', border='#C62828')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch8a(pdf):
    fig, ax = new_page(pdf, title='Chapter 8  —  스테레오 비주얼 오도메트리 (1/2)', chapter='Ch 8 · Stereo VO')
    y = 0.895

    y = section(ax, MARGIN_L, y, '8.1  전체 파이프라인')
    steps8 = [
        ('①', '특징점 검출', '이미지에서 추적할 코너 검출\n(Shi-Tomasi)'),
        ('②', 'LK 시간 추적', '이전→현재 프레임 특징점 추적\n(Lucas-Kanade)'),
        ('③', '스테레오 매칭', '왼쪽→오른쪽 특징점 대응\n(에피폴라 제약)'),
        ('④', '삼각측량', '스테레오 대응으로 metric 3D 좌표\n복원'),
        ('⑤', 'PnP + RANSAC', '3D↔2D 대응에서 상대 pose\n(R, t) 추정'),
    ]
    bx_ = MARGIN_L + 0.005
    for num, title_s, desc in steps8:
        bw_ = 0.160
        ax.add_patch(patches.FancyBboxPatch(
            (bx_, y - 0.080), bw_, 0.080, boxstyle='round,pad=0.004',
            facecolor='#E3F2FD', edgecolor=BLUE, linewidth=0.8,
            transform=ax.transAxes))
        ax.text(bx_ + bw_/2, y - 0.010, num,
                fontsize=12, color=BLUE, ha='center', va='top',
                transform=ax.transAxes, fontfamily='DejaVu Sans')
        ax.text(bx_ + bw_/2, y - 0.030, title_s,
                fontproperties=KR_B, fontsize=8.5, color='#0D47A1',
                ha='center', va='top', transform=ax.transAxes)
        for di, dl in enumerate(desc.split('\n')):
            ax.text(bx_ + bw_/2, y - 0.050 - di*0.016, dl,
                    fontproperties=KR, fontsize=7.2, color=GRAY,
                    ha='center', va='top', transform=ax.transAxes)
        if bx_ + bw_ + 0.014 < 0.97:
            ax.annotate('', xy=(bx_ + bw_ + 0.011, y - 0.040),
                        xytext=(bx_ + bw_, y - 0.040),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2))
        bx_ += bw_ + 0.015
    y -= 0.095

    y = section(ax, MARGIN_L, y, '8.2  특징점 검출 — Shi-Tomasi Corner')
    y = para(ax, MARGIN_L, y, [
        '이미지에서 추적하기 좋은 코너(Corner)를 최대 N개 검출합니다.',
        '코너는 x, y 방향 모두로 명확한 기울기가 있어서 LK 추적이 안정적입니다.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$M = \sum_{patch}  \left( I_x^2,\; I_xI_y;\; I_xI_y,\; I_y^2 \right)_{2\times2}$   (구조 텐서)',
                r'점수 = $\min(\lambda_1, \lambda_2)$   (두 고유값 중 작은 것이 크면 코너)'],
               label='Shi-Tomasi')
    y -= 0.005

    y = section(ax, MARGIN_L, y, '8.3  Lucas-Kanade 광학 흐름 (Optical Flow)')
    y = para(ax, MARGIN_L, y, [
        '이전 프레임의 특징점 위치에서 현재 프레임의 대응 위치를 반복적으로 추정합니다.',
        '가정: 작은 패치 내의 픽셀들은 같은 속도로 움직인다 (Aperture 가정).',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$I(x,y,t) = I(x+u,\,y+v,\,t+dt)$   (밝기 보존 가정)',
                r'$I_x \cdot u + I_y \cdot v = -I_t$   (광학 흐름 방정식)',
                r'피라미드 LK:  큰 해상도 → 작은 해상도 순으로 반복 수렴 (큰 이동 처리)'],
               label='Lucas-Kanade')

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# LK 시간 추적 (이전→현재 프레임)',
        'curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(',
        '    prev_gray,      # 이전 프레임',
        '    curr_gray,      # 현재 프레임',
        '    prev_pts,       # 이전 특징점 위치 (N, 1, 2)',
        '    None,',
        '    winSize=(21, 21),   # 패치 크기',
        '    maxLevel=3,         # 피라미드 레벨',
        '    criteria=(TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 30, 0.01)',
        ')',
        'ok = (status.ravel() == 1)  # 성공한 추적만 선택',
        'tracked_pts = curr_pts[ok]',
    ], title='Pseudocode — Lucas-Kanade Tracking')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch8b(pdf):
    fig, ax = new_page(pdf, title='Chapter 8  —  스테레오 비주얼 오도메트리 (2/2)', chapter='Ch 8 · Stereo VO')
    y = 0.895

    y = section(ax, MARGIN_L, y, '8.4  스테레오 삼각측량 (Triangulation)')
    y = para(ax, MARGIN_L, y, [
        '같은 3D 점 P가 왼쪽 카메라(u_l)와 오른쪽 카메라(u_r)에 각각 투영됩니다.',
        '두 카메라 레이의 교점이 3D 점입니다. 스테레오 베이스라인 b = 6.4 cm에서',
        '시차(Disparity) d = u_l − u_r 로 깊이를 복원합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$Z = \frac{f \cdot b}{d}$   (깊이 = 초점거리 × 베이스라인 / 시차)',
                r'$\sigma_Z \approx \frac{Z^2}{f \cdot b}$   (깊이 오차 ∝ 깊이²)',
                r'T265에서 Z=2m: $\sigma_Z \approx \frac{4}{287\times0.064} \approx 22$ cm/pixel'],
               label='스테레오 깊이')
    y = para(ax, MARGIN_L, y, [
        '따라서 깊이 필터 z_min=0.15m, z_max=2.0m 로 믿을 수 있는 범위만 사용합니다.',
        '에피폴라 제약: T265 카메라가 거의 평행이므로 세로 시차 |Δy| < 3 px 조건으로 아웃라이어 제거.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '8.5  PnP — 3D-2D 대응에서 Pose 추정')
    y = para(ax, MARGIN_L, y, [
        '이전 프레임에서 삼각측량한 3D 점들이 현재 프레임의 어느 픽셀에 투영되는지 알 때,',
        '그 대응을 이용해 카메라 pose (R, t)를 추정합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'주어진: $\{(\mathbf{P}_i^{3D},\; \mathbf{p}_i^{2D})\}_{i=1}^N$   (3D 점, 2D 투영)',
                r'구하기: $(\mathbf{R},\,\mathbf{t})$ 최소화   $\sum_i\|\mathbf{p}_i^{2D} - \pi(\mathbf{R}\mathbf{P}_i+\mathbf{t})\|^2$'],
               label='PnP 문제')

    y = subsection(ax, MARGIN_L, y, 'RANSAC — 아웃라이어에 강인한 PnP')
    y = para(ax, MARGIN_L, y, [
        'RANSAC은 무작위 소집합(Minimal Sample)으로 모델을 추정하고, 이 모델을 지지하는',
        '점 개수(Inlier)가 가장 많은 모델을 선택합니다. 잘못 추적된 특징점(아웃라이어)에',
        '강인합니다.',
    ])

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# PnP + RANSAC',
        'def pnp(pts3d, pts2d, K):',
        '    ok, rvec, tvec, inliers = cv2.solvePnPRansac(',
        '        pts3d, pts2d, K, None,',
        '        iterationsCount=100,',
        '        reprojectionError=2.0,  confidence=0.999,',
        '        flags=SOLVEPNP_ITERATIVE)',
        '    if not ok or len(inliers) < 12: return None',
        '    R, _ = cv2.Rodrigues(rvec)',
        '    t = tvec.ravel()',
        '    if norm(t) > 0.30: return None  # 30cm 초과 → 아웃라이어',
        '    return R, t, inliers',
    ], title='Pseudocode — PnP + RANSAC')

    y = section(ax, MARGIN_L, y, '8.6  좌표계 변환 (핵심 주의사항)')
    y = para(ax, MARGIN_L, y, [
        'PnP는 pts3d가 모두 같은 좌표계(이전 카메라 프레임)에 있어야 올바른 결과를 냅니다.',
        'PnP 결과: P_curr = R @ P_prev + t  로 변환 후 저장 → 새 스테레오 점과 같은 좌표계.',
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch9(pdf):
    fig, ax = new_page(pdf, title='Chapter 9  —  EKF 측정 업데이트와 게이팅', chapter='Ch 9 · Update')
    y = 0.895

    y = section(ax, MARGIN_L, y, '9.1  속도 측정 모델')
    y = para(ax, MARGIN_L, y, [
        'PnP로부터 얻은 상대 이동 t_metric (이전 카메라 좌표계)을 세계 좌표 속도로 변환합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$\Delta\mathbf{p}_W = \mathbf{R}_{world}\,\mathbf{t}_{metric}$   (세계 좌표 변위)',
                r'$\mathbf{v}_{vis} = \Delta\mathbf{p}_W / dt$   (시각 속도 추정치)',
                r'측정 방정식: $\mathbf{z} = \mathbf{v}_{vis} = \mathbf{H}\,\mathbf{x} + \mathbf{n}$'],
               label='속도 측정 모델')
    y = para(ax, MARGIN_L, y, [
        '측정 행렬 H (3×16): H[0:3, 3:6] = I_3 (속도 성분만 관측).',
        '측정 노이즈: R_vel = diag([0.3², 0.3², 0.3²]) m²/s²  (시각 속도 추정 오차 ~0.3 m/s).',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '9.2  칼만 게인과 업데이트')
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}_{vel}$   (혁신 공분산)',
                r'$\mathbf{K} = \mathbf{P}\mathbf{H}^\top\mathbf{S}^{-1}$   (칼만 게인)',
                r'$\mathbf{y} = \mathbf{v}_{vis} - \hat{\mathbf{v}}$   (혁신 = 측정 - 예측)',
                r'$\mathbf{x} \leftarrow \mathbf{x} + \mathbf{K}\mathbf{y}$   (상태 업데이트)',
                r'$\mathbf{P} \leftarrow (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I}-\mathbf{K}\mathbf{H})^\top + \mathbf{K}\mathbf{R}_{vel}\mathbf{K}^\top$   (Joseph 형식)'],
               label='EKF 업데이트 공식', size=9.5)
    y -= 0.005

    y = section(ax, MARGIN_L, y, '9.3  Mahalanobis 거리 게이팅 — 점프 방지')
    y = para(ax, MARGIN_L, y, [
        'PnP가 가끔 잘못된 결과를 낼 때(예: 큰 backward velocity), 이 값을 그대로',
        'EKF에 적용하면 position이 갑자기 이전 위치로 점프합니다. 통계적 게이팅으로 이를 차단합니다:',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'$d^2 = \mathbf{y}^\top \mathbf{S}^{-1} \mathbf{y}$   (Mahalanobis 거리 제곱)',
                r'$d^2 > \chi^2_{3,\,99.9\%} = 16.27 \Rightarrow$ 업데이트 거부 (아웃라이어)',
                r'$d^2 \leq 16.27 \Rightarrow$ 업데이트 수용'],
               label='게이팅')
    y = para(ax, MARGIN_L, y, [
        'χ²(3, 99.9%) = 16.27: 3자유도(속도 x,y,z) 측정에서 통계적으로 가능한 범위 99.9% 내.',
        '게이팅은 S의 크기(현재 불확실도)를 고려하므로, P가 작을 때 같은 혁신이 더 강하게 차단됩니다.',
    ])

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# EKF 시각 업데이트 + Mahalanobis 게이팅',
        'def update_visual(x, P, v_vis, R_vel, gate=16.0):',
        '    H = zeros((3, 16)); H[0:3, 3:6] = eye(3)  # 속도 관측',
        '    y = v_vis - x[3:6]                          # 혁신',
        '    S = H @ P @ H.T + R_vel                     # 혁신 공분산',
        '',
        '    # Mahalanobis 거리 게이팅',
        '    md2 = y @ solve(S, y)',
        '    if md2 > gate:',
        '        return x, P, False  # 아웃라이어 → 거부',
        '',
        '    K = P @ H.T @ inv(S)                        # 칼만 게인',
        '    dx = K @ y',
        '    x_new = x + dx',
        '    x_new[6:10] = normalize(x_new[6:10])        # 쿼터니언 재정규화',
        '',
        '    I_KH = eye(16) - K @ H',
        '    P_new = I_KH @ P @ I_KH.T + K @ R_vel @ K.T  # Joseph 형식',
        '    return x_new, P_new, True',
    ], title='Pseudocode — EKF Visual Update with Gating')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch10(pdf):
    fig, ax = new_page(pdf, title='Chapter 10  —  ZUPT (Zero Velocity Update)', chapter='Ch 10 · ZUPT')
    y = 0.895

    y = section(ax, MARGIN_L, y, '10.1  이중 적분 드리프트 문제')
    y = para(ax, MARGIN_L, y, [
        'IMU 가속도를 이중 적분하면 위치를 얻지만, 바이어스와 노이즈가 누적되어 발산합니다.',
        '카메라 업데이트가 없는 구간(폐색, 조명 불량 등)에서 EKF는 IMU 예측만 수행하게 되어',
        '속도 오차가 쌓입니다. 디바이스가 실제로 정지해 있다면 이를 이용해 강제 보정합니다.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '10.2  ZUPT 원리')
    y = para(ax, MARGIN_L, y, [
        '정지 조건 감지: IMU 측정값에서 디바이스가 정지했음을 판단합니다.',
        '감지 시: 속도 = 0 이라는 측정값(z = 0)을 EKF에 주입합니다.',
        '효과: 속도 오차 누적을 주기적으로 초기화하여 위치 드리프트 억제.',
    ])
    y = eq_box(ax, MARGIN_L + 0.05, y, 0.77,
               [r'정지 조건: $|\,|\tilde{\mathbf{a}}| - 9.81| < 0.8$ m/s²  AND  $|\tilde{\mathbf{\omega}}| < 0.08$ rad/s',
                r'ZUPT 측정: $\mathbf{z}_{zupt} = \mathbf{0} = \mathbf{H}\,\mathbf{x} + \mathbf{n}_{zupt}$',
                r'$\mathbf{R}_{zupt} = \sigma_{zupt}^2\mathbf{I}_3,\quad \sigma_{zupt} = 0.01$ m/s  (tight)'],
               label='ZUPT')
    y = para(ax, MARGIN_L, y, [
        '|ã| ≈ 9.81: 가속도 크기가 중력과 같다 = 정지 (추가 운동 가속도 없음).',
        '|w_raw| ≈ 0: 회전이 없다. 두 조건 모두 만족 시 ZUPT를 적용합니다.',
    ])
    y -= 0.005

    y = section(ax, MARGIN_L, y, '10.3  ZUPT EKF 업데이트')
    y = eq_box(ax, MARGIN_L + 0.03, y, 0.81,
               [r'$\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}_{zupt}$',
                r'$\mathbf{K} = \mathbf{P}\mathbf{H}^\top\mathbf{S}^{-1}$',
                r'$\mathbf{y} = \mathbf{0} - \hat{\mathbf{v}} = -\hat{\mathbf{v}}$   (혁신: 속도를 0으로)',
                r'$\mathbf{x} \leftarrow \mathbf{x} + \mathbf{K}\mathbf{y}$,\quad $\mathbf{P} \leftarrow (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I}-\mathbf{K}\mathbf{H})^\top + \mathbf{K}\mathbf{R}_{zupt}\mathbf{K}^\top$'],
               label='ZUPT 업데이트', size=9.5)

    y = code_block(ax, MARGIN_L, y, 0.87, [
        '# ZUPT 의사코드',
        'def zupt_if_stationary(bundle, ekf):',
        '    a_mag = norm(mean([a for _, a in bundle.accel_samples]))',
        '    g_mag = norm(mean([g for _, g in bundle.gyro_samples]))',
        '',
        '    if abs(a_mag - 9.81) < 0.8 and g_mag < 0.08:',
        '        # 정지 확인 → 속도 = 0 업데이트',
        '        H = zeros((3, 16)); H[0:3, 3:6] = eye(3)',
        '        R_zupt = eye(3) * (0.01 ** 2)   # 매우 타이트한 노이즈',
        '        y = -ekf.x[3:6]                 # 혁신 = 0 - v_current',
        '        S = H @ ekf.P @ H.T + R_zupt',
        '        K = ekf.P @ H.T @ inv(S)',
        '        ekf.x += K @ y',
        '        ekf.x[6:10] = normalize(ekf.x[6:10])',
        '        I_KH = eye(16) - K @ H',
        '        ekf.P = I_KH @ ekf.P @ I_KH.T + K @ R_zupt @ K.T',
    ], title='Pseudocode — ZUPT')

    y = info_box(ax, MARGIN_L, y, 0.87, [
        '순서 중요:  ZUPT는 시각 업데이트 이후 마지막에 수행합니다.',
        '시각 업데이트 전에 ZUPT를 하면 카메라 업데이트가 ZUPT를 덮어써 효과가 없어집니다.',
        '정지 감지 임계값 (0.8 m/s², 0.08 rad/s)은 일반 보행 중 ZUPT가 오발동하지 않도록 설정.',
    ], color='#FFF9C4', border='#F9A825')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_ch11(pdf):
    """전체 시스템 파이프라인 — 최종 정리"""
    fig, ax = new_page(pdf, title='Chapter 11  —  전체 시스템 파이프라인', chapter='Ch 11 · 완성')
    y = 0.895

    y = section(ax, MARGIN_L, y, '11.1  매 프레임 처리 흐름')

    # 전체 흐름 다이어그램 (수직)
    flow_items = [
        ('IMU 샘플 버퍼링\n(200 Hz, 동기화)',              '#1565C0'),
        ('중력 정렬 완료? (1초 대기)',                      '#6A1B9A'),
        ('IMU 적분 → EKF.predict()\n(버퍼 내 모든 샘플)', '#1976D2'),
        ('스테레오 이미지 도착? (30 Hz)',                  '#2E7D32'),
        ('피시아이 왜곡 보정\n(Kannala-Brandt → Pinhole)', '#388E3C'),
        ('LK 시간 추적 + PnP\n(이전 3D 점 → 현재 Pose)', '#E65100'),
        ('새 특징점 검출 + 삼각측량\n(스테레오 깊이 복원)', '#F57F17'),
        ('좌표계 변환 후 3D 점 합산',                      '#BF360C'),
        ('EKF.update_visual() + 게이팅',                  '#C62828'),
        ('ZUPT (정지 감지 시)',                             '#4A148C'),
        ('위치 / 자세 / 속도 출력',                        '#1B5E20'),
    ]
    bx_ = 0.12
    by_ = y - 0.032
    bw_ = 0.76
    bh_ = 0.044
    for i, (label, color) in enumerate(flow_items):
        ax.add_patch(patches.FancyBboxPatch(
            (bx_, by_), bw_, bh_, boxstyle='round,pad=0.004',
            facecolor=color, edgecolor='white', linewidth=0.5,
            transform=ax.transAxes, alpha=0.9))
        lines = label.split('\n')
        cy_inner = by_ + bh_/2 + (len(lines)-1)*0.008
        for line in lines:
            ax.text(bx_ + bw_/2, cy_inner, line,
                    fontproperties=KR, fontsize=8, color='white',
                    ha='center', va='center', transform=ax.transAxes)
            cy_inner -= 0.016
        if i < len(flow_items) - 1:
            ax.annotate('', xy=(bx_ + bw_/2, by_ - 0.007),
                        xytext=(bx_ + bw_/2, by_),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.2))
        by_ -= bh_ + 0.012
    y = by_ - 0.010

    y = section(ax, MARGIN_L, y, '11.2  파라미터 정리')
    params = [
        ('σ_a = 0.01 m/s²/√Hz',  '가속도 노이즈 — T265 스펙 기준'),
        ('σ_g = 0.002 rad/s/√Hz', '자이로 노이즈'),
        ('r_vel = 0.3 m/s',       'EKF 시각 속도 측정 노이즈 (std)'),
        ('χ² gate = 16.0',        'Mahalanobis 게이팅 (99.9%)'),
        ('max_corners = 250',     '특징점 최대 개수 (LK/PnP 성능 상한)'),
        ('z_min/max = 0.15~2.0 m','유효 삼각측량 깊이 범위'),
    ]
    ax.add_patch(patches.FancyBboxPatch(
        (MARGIN_L + 0.02, y - 0.090), 0.83, 0.086, boxstyle='round,pad=0.005',
        facecolor='#E3F2FD', edgecolor=BLUE, linewidth=0.8,
        transform=ax.transAxes))
    py_ = y - 0.012
    for param, note in params:
        ax.text(MARGIN_L + 0.04, py_, param,
                fontproperties=MONO, fontsize=8, color='#0D47A1',
                va='top', transform=ax.transAxes)
        ax.text(MARGIN_L + 0.35, py_, note,
                fontproperties=KR, fontsize=8, color=GRAY,
                va='top', transform=ax.transAxes)
        py_ -= 0.014
    y -= 0.100

    y = info_box(ax, MARGIN_L, y, 0.87, [
        '핵심 설계 원칙:',
        '① IMU는 예측, 카메라는 보정 — 두 센서를 EKF로 융합',
        '② 모든 3D 점은 같은 좌표계(이전 카메라 프레임)에 있어야 PnP가 올바름',
        '③ Mahalanobis 게이팅으로 잘못된 시각 업데이트 차단',
        '④ ZUPT로 정지 시 이중 적분 드리프트 억제 (시각 업데이트 다음에 실행)',
    ], color='#E8F5E9', border='#2E7D32')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    out_path = '/home/shim/Project/Study/VIO/T265/VIO_Theory_Guide.pdf'
    print(f'Generating {out_path} ...')
    with PdfPages(out_path) as pdf:
        page_cover(pdf)
        page_ch1(pdf)
        page_ch2(pdf)
        page_ch3(pdf)
        page_ch4(pdf)
        page_ch5(pdf)
        page_ch6(pdf)
        page_ch7(pdf)
        page_ch8a(pdf)
        page_ch8b(pdf)
        page_ch9(pdf)
        page_ch10(pdf)
        page_ch11(pdf)

        # PDF 메타데이터
        d = pdf.infodict()
        d['Title'] = 'Visual-Inertial Odometry: 이론과 구현'
        d['Author'] = 'T265 VIO Study'
        d['Subject'] = 'EKF VIO Theory — T265 Stereo Fisheye'
        d['Keywords'] = 'VIO EKF IMU Stereo Kalman Filter Quaternion'

    print(f'Done → {out_path}')


if __name__ == '__main__':
    main()
