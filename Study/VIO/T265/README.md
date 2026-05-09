# T265 Stereo VIO — EKF-based Visual-Inertial Odometry

Intel RealSense T265 카메라를 이용한 Stereo Visual-Inertial Odometry 구현입니다.  
IMU와 스테레오 카메라를 **Extended Kalman Filter**로 융합하여 6-DOF 위치/자세를 실시간으로 추정합니다.

---

## 시연 영상

https://github.com/user-attachments/assets/2089b21b-1e76-410b-9ac8-2402c1699931

> 왼쪽: 피시아이 카메라 + LK 특징점 추적 / 오른쪽: 깊이 맵 + EKF 추적 궤적

---

## 아키텍처

```
T265Reader ──┬── accel/gyro ──► ImuIntegrator ──► EKF.predict()
             │                                          │
             └── stereo frames ──► StereoVisualTracker  │
                                       │                ▼
                                  R, t_metric ──► EKF.update_visual()
                                                        │
                                                  EKF.update_zupt()  ← 정지 감지 시
```

### 모듈 구성

| 파일 | 역할 |
|------|------|
| `ekf.py` | 16차원 EKF (위치·속도·쿼터니언·가속도바이어스·자이로바이어스) |
| `visual_tracker.py` | Kannala-Brandt 왜곡 보정 → LK 추적 → PnP → 스테레오 삼각측량 |
| `imu_integrator.py` | IMU 샘플 버퍼링 및 EKF predict 일괄 적용 |
| `vio_tracker.py` | 중력 정렬 초기화, ZUPT 감지, 전체 파이프라인 조율 |
| `t265_reader.py` | pyrealsense2 인터페이스 + 하드웨어 없을 때 mock 모드 |
| `visualizer.py` | OpenCV 실시간 오버레이 시각화 |
| `main.py` | 진입점 |

---

## EKF 상태 벡터 (16차원)

```
x = [ p(3)  v(3)  q(4)  b_a(3)  b_g(3) ]
      위치   속도  쿼터니언  가속도바이어스  자이로바이어스
```

### 핵심 설계 포인트

- **스케일 해결**: 스테레오 베이스라인(6.4 cm)으로 metric scale 직접 획득 → scale 추정 불필요
- **좌표계 일관성**: PnP 인라이어 3D 점을 `P_curr = R @ P_prev + t`로 변환 후 저장 (좌표계 혼용 버그 방지)
- **Mahalanobis 게이팅**: χ²(3-DOF, 99.9%) = 16.27 임계값으로 PnP 이상치 차단 → 위치 점프 방지
- **ZUPT**: 정지 상태(|a|≈9.81 m/s², |ω|<0.08 rad/s) 감지 시 v=0 업데이트로 이중 적분 드리프트 억제
- **중력 초기화**: 1초간 정지 가속도 평균으로 초기 자세 정렬 + 자이로 바이어스 사전 추정

### 프로세스 노이즈 (T265 IMU 스펙 기반)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| σ_a | 0.01 m/s²/√Hz | 가속도 노이즈 |
| σ_g | 0.002 rad/s/√Hz | 자이로 노이즈 |
| σ_ba | 5×10⁻⁵ | 가속도 바이어스 랜덤워크 |
| σ_bg | 1×10⁻⁵ | 자이로 바이어스 랜덤워크 |
| R_vel | 0.3² m²/s² | 시각 속도 측정 노이즈 |

---

## 스테레오 VO 파이프라인

```
1. Kannala-Brandt fisheye 보정 → pinhole 이미지
2. 이전 프레임 특징점 → 현재 프레임 LK 광류 추적
3. solvePnPRansac (3D↔2D) → metric R, t 추정
4. 현재 프레임 새 특징점 검출 → 오른쪽 이미지로 LK 매칭
5. 스테레오 삼각측량 → metric 3D 랜드마크
6. 깊이 필터: z ∈ [0.15 m, 2.0 m] (T265 스테레오 정확도 범위)
```

---

## 설치 및 실행

### 1. librealsense SDK (T265 카메라 사용 시)

T265는 Intel이 공식 지원을 종료했기 때문에 최신 Ubuntu에서 빌드하려면 패치가 필요합니다.  
Ubuntu 24.04 / GCC 13 환경에서 빌드된 **사전 빌드 바이너리**를 아래 레포에서 제공합니다:

> **[Naturalium/librealsense-t265](https://github.com/Naturalium/librealsense-t265)**  
> — pybind11 v2.11.1 업데이트, GCC 13 컴파일 오류 수정 적용  
> — [Releases](https://github.com/Naturalium/librealsense-t265/releases/tag/v2.51.1)에서 `librealsense2.so.2.51.1` 다운로드 후 설치

```bash
# 바이너리 설치
sudo cp librealsense2.so.2.51.1 /usr/local/lib/
sudo ln -sf /usr/local/lib/librealsense2.so.2.51.1 /usr/local/lib/librealsense2.so.2
sudo ln -sf /usr/local/lib/librealsense2.so.2 /usr/local/lib/librealsense2.so
sudo ldconfig

# T265 USB 권한 설정 (librealsense-t265 소스 클론 후)
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2. Python 패키지

```bash
pip install numpy>=1.24 opencv-python>=4.8
pip install pyrealsense2>=2.54  # T265 실제 카메라 사용 시
```

### 실행

```bash
# T265 카메라 연결 후 (시각화 포함)
python main.py

# 헤드리스 모드 (터미널 출력만)
python main.py --no-vis

# 카메라 없이 mock 데이터로 테스트
python main.py --mock
```

### 교육 문서 생성

```bash
python generate_vio_doc.py
# → VIO_Theory_Guide.pdf 생성 (EKF VIO 이론 + 의사코드 13페이지)
```

---

## 학습 문서

`VIO_Theory_Guide.pdf` — EKF VIO 이론을 학생 관점에서 설명한 13페이지 문서:

- 쿼터니언, 회전 행렬, IMU 모델
- EKF 예측·업데이트 단계 (Jacobian 유도 포함)
- 스테레오 삼각측량, PnP, RANSAC
- Mahalanobis 게이팅, ZUPT
- 전체 파이프라인 의사코드

---

## 하드웨어

- **Intel RealSense T265**: 듀얼 fisheye (Kannala-Brandt, 170° FOV) + BMI055 IMU (200 Hz)
- 스테레오 베이스라인: 64 mm

## 관련 레포지토리

| 레포 | 설명 |
|------|------|
| [Naturalium/librealsense-t265](https://github.com/Naturalium/librealsense-t265) | Ubuntu 24.04 / GCC 13용 librealsense v2.51.1 패치 빌드본 |
| [IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense) | 업스트림 원본 (v2.51.1이 T265 마지막 지원) |
