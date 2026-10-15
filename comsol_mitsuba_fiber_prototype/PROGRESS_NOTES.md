# Mitsuba 3 광섬유 광선추적 파이프라인: 진행 정리 (2026-10-05 ~ 06)

COMSOL FEM(변형 형상) → Mitsuba 3 GPU 광선추적 → Pout/Pin 파이프라인을 만들고 있다.
COMSOL 없이 합성 형상으로 파이프라인을 검증했고, 이제 COMSOL Ray Optics와 비교하려 한다.
마지막 절에 **확인이 필요한 질문**을 모아 두었다.

---

## 1. 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX 5070 (12 GB), driver 595.91 |
| Python | conda env `mitsuba` (Python 3.14) |
| Mitsuba | pip `mitsuba 3.9.1`, `drjit 1.5.0`, variant `cuda_ad_rgb` (OptiX) |
| 기타 | numpy, scipy, trimesh, rtree, PyYAML, polyscope (시각화) |
| 참고 | `mitsuba3/`에 소스 빌드(3.9.0)도 있으나 사용하지 않음. pip 버전에는 polarized variant가 없음 |

---

## 2. 테스트 문제 정의 (사용자 확정 사항)

| 항목 | 값 |
|---|---|
| core / cladding 지름 | 240 µm / 250 µm (cladding 두께 5 µm) |
| 굴절률 (630 nm) | core 1.492, cladding 1.402, 주변 공기 1.000277 (config `ior.air`로 변경 가능) |
| 전체 길이 | 50 mm (중심선 기준) |
| 형상 | 직선 L1 → 180° bend → 직선 L1 (U자). bend 반지름 **중심선 기준 500 µm**. L1 = (50 − π·0.5)/2 = 24.2146 mm |
| 좌표 (mm) | 입력면 중심 (0,0,0), 축 +z. bend 중심 (0.5, 0, L1), xz 평면. 출력면 중심 (1, 0, 0) |
| 광원 | 입력면 1 µm 안쪽 축 위 점광원. **core 유리 내부에서 반각 30°** 원뿔, 입체각 균일(cosθ 균일), 광선당 동일 파워 |
| 측정량 | 반대쪽 끝면에서 core + cladding 영역으로 나오는 파워 합 / 입력 파워 (광선 개수 기준 파워 비율) |
| 광학 모델 | 기하광학, 비편광 Fresnel(Rs, Rp 평균), 흡수 없음, 굴절률 상수(분산 없음) |

---

## 3. 코드 구조 (`comsol_mitsuba_fiber_prototype/`)

```
make_test_fiber.py ─► stl/*.stl ─► prepare_meshes.py ─► prepared/*.ply ─► trace_mitsuba.py ─► Pout/Pin
(COMSOL export 대체)                (용접·접촉 분류)                      (GPU 광선추적)
                                                         view_rays.py (3D 실시간 뷰어)
                                                         check_straight_2d.py (독립 검증)
                                                         estimate_secondary_rays.py (COMSOL 분기 방식 모사)
```

| 파일 | 역할 |
|---|---|
| `make_test_fiber.py` | 중심선을 따라 원을 sweep해 튜브 메쉬 생성 (둘레 720, bend 360분할, 직선 1 mm 간격). 법선은 바깥쪽. `--straight`, `--length`, `--bend-radius` 등 |
| `prepare_meshes.py` | STL 로드 → **꼭짓점 용접** → 바깥 cladding을 silicone 거리로 `cladding_air` / `cladding_contact` 분류 (`--silicone` 생략 가능) |
| `trace_mitsuba.py` | 장면 구성(유전체 경계 3종 + detector 태그) → 광선 발사 → 반복 추적 → 결과 출력 |
| `view_rays.py` | polyscope 3D 뷰어. 각도·광선 수·bend 반지름 슬라이더, 재계산 약 0.6 s, 경로 애니메이션 |
| `check_straight_2d.py` | 직선 광섬유를 2D(자오광선)로 numpy에서 독립 계산 |
| `estimate_secondary_rays.py` | COMSOL식 결정론적 분기(반사·굴절 모두 추적 + threshold)를 Mitsuba 장면에서 모사 |

### 추적 알고리즘 (`trace_mitsuba.py`의 `trace()`)

각 반복에서 모든 광선(10⁶개)을 GPU로 동시에 처리:
1. `scene.ray_intersect` (OptiX)
2. 미충돌 → 손실(escaped)
3. detector 충돌 → 검출. 충돌점 반경 < 120 µm면 core, 아니면 cladding. 출사각으로 유리→공기 Fresnel 투과율 T를 가중치로 기록
4. 그 외 경계 → Mitsuba `dielectric` BSDF로 **확률 R로 반사, 1−R로 굴절** (전반사면 R=1). 광선 수가 늘지 않음
5. 새 광선 원점을 법선 방향으로 `(1 + max|p|) × 2⁻²⁰`만큼 띄움 (자체 구현, 아래 4-1 참고)

출력:
- `transmission`: 출력면 도달 (Fresnel 전)
- `reach_end_core` / `reach_end_cladding`: 위 값의 영역별 분리
- `exit_total`: 공기로 나간 비율 = **Pout/Pin**
- `exit_core` / `exit_cladding`: 위 값의 영역별 분리
- `escaped`: 측면이나 입력단으로 빠져나간 광선 수
- `terminated_at_max_bounces`: 반사 횟수 한도에 걸린 광선 수
- 통계오차 ≈ √(p(1−p)/N)

---

## 4. 발견하고 수정한 문제

### 4-1. Mitsuba `spawn_ray()` 오프셋이 cladding 두께와 비슷함 (결과를 바꾸는 버그)
- 증상: 직선 광섬유에서 출력단 cladding 비율이 Mitsuba 0.0044, 2D 독립 계산 0.0428로 약 10배 차이.
- 원인: Mitsuba의 2차 광선 오프셋은 약 `(1 + max|p|) × 1500 ulp ≈ (1 + max|p|) × 9e-5`. mm 단위에서 z = 50 mm면 약 4.5 µm라 5 µm cladding을 건너뜀. 좌표 크기에 비례하므로 단위를 바꿔도 해결되지 않음.
- 진단: 경로상 n·cos θ_z 불변량은 정확히 보존됐지만, 출력단 근처 cladding 왕복 구간 길이가 이론값 약 15 µm 대비 1.3 µm로 줄어 있었음.
- 수정: 오프셋을 `(1 + max|p|) × 2⁻²⁰`(50 mm에서 약 48 nm)으로 직접 계산 (`simulation.ray_offset_rel`). 2⁻¹⁸~2⁻²²에서 결과가 안정적.

### 4-2. 꼭짓점이 2²⁴개를 넘는 메쉬에서 삼각형이 사라짐
- 증상: 2880×1440 분할 메쉬에서 투과율 0.0003. 반경 방향 광선의 24.5%가 메쉬를 그냥 통과.
- 원인: STL은 삼각형마다 꼭짓점 3개를 따로 저장해 꼭짓점이 2,570만 개. 2²⁴(16.7M)를 넘으면 Mitsuba 3.9.1 / OptiX에서 일부 면이 사라짐 (관측 사실, 내부 원인은 미확인).
- 수정: `prepare_meshes.py`에서 `merge_vertices()` 후 2²⁴ 초과 시 경고. 용접 후 430만 개로 정상 동작.

### 4-3. 원본 스크립트가 Mitsuba 3.9에서 동작하지 않던 부분
- `mi.is_cuda` 없음, `mi.RayFlags.Default` 없음, `ray_mask` 인자 없음, area emitter의 `twosided` 속성 없음.
- detector 판정을 `emitter.eval() > 0`(단면 emitter라 뒷면 충돌을 놓침)에서 `si.shape.is_emitter()`로 변경.
- 루프에서 `active`만 평가하던 구조라 계산 그래프가 계속 커져 10분 이상 걸림 → 매 반복 전체 상태를 `dr.eval`하도록 바꿔 약 0.3 s.
- 접촉 면이 없을 때 `cladding_contact.ply`가 없어서 실패 → 해당 메쉬를 선택 항목으로 처리.

---

## 5. 검증 결과 (광선 10⁶개)

### 5-1. 직선 50 mm, 반각 30°: Mitsuba vs 독립 2D 계산

| 항목 | 2D 기준 (2만 개) | Mitsuba |
|---|---|---|
| 출력면 도달, core | 0.9573 | 0.9566 |
| 출력면 도달, cladding | 0.0428 (±0.0014) | 0.0434 |
| Pout/Pin | 0.95798 | 0.95801 |
| 손실 | 0 | 4e-6 |

통계오차 범위 안에서 일치. 메쉬 분할(360~4320)에 무관.

### 5-2. U자 bend (R = 0.5 mm), 반각 30°: 본 결과

| 항목 | 값 |
|---|---|
| 출력면 도달 | 0.9002 |
| 그중 core / cladding | 0.8557 / 0.0445 |
| **Pout/Pin** | **0.8288** (core 0.7861, cladding 0.0427) |
| 측면 손실 | 약 0.100 (주로 bend 입구 바깥쪽) |
| 평균 상호작용 수 | 약 208회 |
| 계산 시간 | 약 0.5 s |

- 수렴 확인: 분할 360/180 → 2880/1440에서 Pout/Pin 0.8288~0.8290, 오프셋 2⁻¹⁸~2⁻²², 시드 변경 시 ±0.0006.
- bend 구간은 해석해가 없어 수렴성으로만 확인함.

### 5-3. 민감도 (Pout/Pin)

| 조건 | 직선 | U자 R=0.5 mm |
|---|---|---|
| 반각 30° (core 내부) | 0.958 | 0.829 |
| 반각 15° | 0.961 | 0.912 |
| 반각 19.58° (공기 중 30°에 해당) | 0.961 | 0.897 |
| bend 반지름 1 mm / 3 mm (30°) | — | 0.946 / 0.957 |

### 5-4. 길이 20 mm로 재실행 (사용자 테스트)
`--length 20`으로 생성한 결과: 출력면 도달 0.8989, Pout/Pin 0.8279, 평균 상호작용 85회. bend 손실이 지배적이라 50 mm와 비슷하게 나옴.

---

## 6. COMSOL Ray Optics와 비교 계획

### 6-1. 동일하게 맞출 설정
- **형상:** core와 cladding을 별도 도메인으로 두고, 광섬유를 감싸는 공기 도메인 추가 (바깥 경계는 Wall: Disappear). "Use geometry normals for ray-boundary interactions" 사용.
- **재료:** 굴절률 실수부만 (1.492 / 1.402 / 1.000277).
- **물리:** Intensity computation = Compute power. Ray Tracing study, 광학 경로 약 150 mm.
- **광원:** Release from Point (0, 0, 0.001 mm), Conical, 축 +z, 반각 30°. **입체각 균일 분포인지 cosθ 히스토그램으로 확인** (Hexapolar는 피하기).
- **입력단:** Mitsuba는 열린 끝(반사 없음)이므로 입력 끝면은 Wall: Disappear 권장.
- **측정 A (Fresnel 전):** 출력 끝면을 Wall: Freeze로 두고 Ray Evaluation(Sum)으로 파워를 합산.
  ```
  core:     gop.Q*(abs(qz)<1e-4[mm])*(sqrt((qx-1[mm])^2+qy^2)<0.12[mm])
  cladding: gop.Q*(abs(qz)<1e-4[mm])*(sqrt((qx-1[mm])^2+qy^2)>=0.12[mm])
  ```
- **측정 B (Pout/Pin):** 출력 끝면은 Material Discontinuity로 두되 반사광은 생성하지 않게 하고(Mitsuba는 끝면 반사분을 버림), 끝면에 바짝 붙인 검출면에서 Freeze. U자라 입력단과 출력단이 1 mm 간격이므로, 새어 나온 광선이 섞이지 않게 주의.
- **에너지 보존:** 도달 + 측면 손실 + threshold로 버려진 파워 + 미완료 ≈ 1인지 확인.

### 6-2. 핵심 문제: COMSOL 결정론적 분기는 이 형상에서 수렴하지 않음

`estimate_secondary_rays.py`로 COMSOL 방식을 모사한 결과 (1차 광선 2,000개, threshold는 1차 광선 1개 파워 대비):

| 경우 | threshold | 2차 광선 / 1차 광선 | 출력면 도달 | threshold로 버려진 파워 | Mitsuba (확률적) |
|---|---|---|---|---|---|
| 직선 30° | 1e-4 | 1,851 | 0.446 | 55% | 1.000 |
| U자 30° | 1e-3 | 212 | 0.273 | 70% | 0.900 |
| U자 30° | 1e-4 | 2,455 | 0.295 | 67% | 0.900 |
| U자 15° | 1e-4 | 1,401 | 0.580 | 38% | 0.966 |
| 직선 15° / 19.58° | — | 0 | 1.000 | 0% | 1.000 |

- 원인: θ > 약 20° 광선은 core/cladding 경계에서 매번 R ≈ 5% 부분 반사가 일어남. 50 mm 동안 수백 번 분기하면서 파워가 지수적으로 많은 가지로 쪼개지고, 대부분 threshold 아래로 내려감. threshold를 10배 낮춰도 버려지는 파워가 70%에서 67%로만 줄어듦.
- bend가 있으면 15° 원뿔도 bend에서 각도가 커져 같은 문제가 생김.
- 단 한 번의 추적에서 분기 계산의 파워 보존은 확인함 (각 단계 합계 = 1).

### 6-3. 현재 권장 비교 방식
1. **직선, 반각 15° 또는 19.58°**: 분기 없음. secondary rays 약 1,000이면 충분. Mitsuba 기준: 도달 1.0000, Pout/Pin 0.9609 (15°) / 0.9606 (19.58°). 형상, 광원, 검출 설정 확인용.
2. **U자 30°를 같은 알고리즘으로 비교**: COMSOL threshold = 1e-3 × (총 광원 파워 / N), Maximum number of secondary rays ≈ 250 × N (N = 1,000~2,000). 모사 결과(도달 0.273, core 0.271, cladding 0.003, 측면 0.031, 버려진 파워 0.696)와 비교. 맞으면 Mitsuba의 0.900이 같은 물리를 수렴시킨 값이라고 볼 수 있음. 단, COMSOL의 정확한 분기 규칙(투과 쪽 광선에도 threshold를 적용하는지 등)을 모르기 때문에 몇 % 차이는 가능.

---

## 7. 확인이 필요한 질문

### COMSOL 관련
1. COMSOL Ray Optics의 Material Discontinuity에 반사와 굴절 중 하나를 **확률적으로(Monte Carlo) 고르는 옵션**이 있는가? 있다면 2차 광선 없이 30° bend를 직접 비교할 수 있다.
2. "Maximum number of secondary rays"는 전체 시뮬레이션의 누적 총량인가, 동시에 살아 있는 광선 수인가? 메모리 비용은 어느 정도인가?
3. threshold(intensity / power)는 반사된 2차 광선 생성에만 적용되는가, 아니면 기존 광선이 threshold 아래로 떨어지면 종료시키는가? 버려진 파워를 결과로 확인하는 방법은?
4. Release from Point의 Conical 분포 옵션 중 **입체각 균일 + 광선당 동일 파워**에 해당하는 것은 무엇인가?
5. 외부(비메쉬) 영역의 굴절률 설정만으로 cladding↔공기 경계의 Fresnel 반사와 전반사가 정확히 처리되는가, 아니면 공기 도메인을 메쉬해야 하는가?
6. 끝면에서 반사광을 생성하지 않고 굴절광만 내보내는 설정(측정 B용)과, Freeze된 광선 파워를 경계별로 합산하는 가장 좋은 방법(Ray Evaluation, Deposited Ray Power 등)은?
7. 5 µm cladding이 있는 50 mm 형상의 메쉬 권장 사항은? geometry normals를 쓸 때도 메쉬 해상도가 결과에 영향을 주는가?

### 물리 / 모델 관련
8. 비편광 근사의 오차: Mitsuba는 매 상호작용마다 (Rs+Rp)/2를 쓰고, COMSOL은 Stokes 벡터를 추적한다. 수백 번의 전반사(s/p 위상차)와 부분 반사 뒤에 얼마나 차이가 날 수 있는가? 편광을 추적해야 하는 경우인가?
9. 이 문제에서 확률적 분기(Mitsuba)와 결정론적 분기(COMSOL, threshold → 0)는 같은 기댓값으로 수렴하는가? 30° bend에서 Mitsuba 0.900이 물리적으로 맞다고 볼 근거가 충분한가?
10. 실제 측정과 비교할 때 기하광학이 충분한가? (240 µm core, 630 nm, V-number가 매우 큼.) R/a = 4인 급격한 bend에서 파동 효과(누설 모드, 터널링 광선)를 고려해야 하는가?
11. 입력단 Fresnel 반사(역방향 광선이 입력면에서 다시 반사되는 효과)와 재료 흡수를 넣어야 하는가?

### Mitsuba 관련
12. 2²⁴ 꼭짓점 초과 시 면이 사라지는 현상이 Mitsuba / OptiX의 알려진 제한인가? (pip 3.9.1, cuda_ad_rgb에서 관측)
13. `ray_offset_rel = 2⁻²⁰`이 자기 교차(self-intersection) 없이 안전한 범위인가? 더 원칙적인 방법(예: Wächter & Binder의 robust offset)이 있는가?

---

## 8. 재현 명령

```bash
conda activate mitsuba
cd ~/hajoon/mitsuba/comsol_mitsuba_fiber_prototype

# 형상 → 메쉬 → 추적
python make_test_fiber.py --out my_test/stl
python prepare_meshes.py --core my_test/stl/core_boundary.stl \
  --cladding my_test/stl/cladding_outer.stl --detector my_test/stl/detector.stl --out my_test/prepared
cp test_bent_fiber/config.yaml my_test/config.yaml
python trace_mitsuba.py --config my_test/config.yaml --rays 1000000

# 검증
python check_straight_2d.py
python trace_mitsuba.py --config test_straight_fiber/config.yaml --rays 1000000

# COMSOL 분기 방식 모사 (secondary rays 추정)
python estimate_secondary_rays.py --config test_bent_fiber/config.yaml --primary 2000 --thresholds 1e-3 1e-4

# 3D 뷰어
python view_rays.py --config test_bent_fiber/config.yaml --live-bend
```

---

## 9. 추가: 실험과의 괴리와 cladding 바깥면 민감도 (2026-10-06)

사용자 관찰: 실제로 R = 500 µm로 구부리면 손실이 크게 보이는데, 시뮬레이션은 15°에서 Pout/Pin 0.91.

같은 U자 형상에서 cladding 바깥 경계만 바꿔 본 결과:

| 발사 조건 | 맨 광섬유 (cladding↔공기 경계 완벽) | cladding을 따라가는 빛 제거 (바깥 n = 1.49) |
|---|---|---|
| 15°, 점광원 | 0.912 | 0.077 |
| 15°, core 전체에서 발사 | 0.899 | 0.174 |
| 30°, 점광원 | 0.829 | 0.022 |

- R/a ≈ 4에서는 core↔cladding 전반사(70°)로 빛을 가둘 수 없어 거의 모든 빛이 core를 벗어난다. bend 통과는 cladding↔공기 전반사(45.5°)에 의존한다.
- 따라서 결과가 바깥면의 상태(거칠기, 오염, 접촉)에 매우 민감하다.
- 광섬유는 Mitsubishi ESKA SK-10(240/250 µm, PMMA, NA 0.5)로 추정된다. 사양상 최소 bend 반경은 5 mm(90° 굽힘에서 손실 0.5 dB 미만)이고, 0.5 mm면 바깥 표면 변형률이 약 25%라 소성 변형이나 crazing이 생길 수 있다.
- 참고 문헌: Sci. Rep. 2020 "Experimental determination and ray-tracing simulation of bending losses in melt-spun POF" (경계면 거칠기 산란 모델 포함), "Dependence of bending losses on cladding thickness in POF".

## 10. 추가: 실리콘(n = 1.41) 접촉 깊이 스윕

상세 내용은 `contact_sweep/RESULTS.md`에 있다. bend 꼭대기 위에서 실리콘 블록을 d만큼 겹치게 하고(광섬유 변형 없음, 완전 접촉 가정) 계산한 Pout/Pin:

| d [µm] | 0 | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| 15° | 0.912 | 0.675 | 0.314 | 0.156 | 0.124 | 0.103 |
| 30° | 0.829 | 0.542 | 0.235 | 0.094 | 0.066 | 0.053 |

### 추가로 확인할 질문
14. 실리콘의 630 nm 실제 굴절률은 얼마인가? n_실리콘 > n_cladding(1.402) 여부가 결과를 좌우한다.
15. 접촉 시 광섬유 변형과 실제 접촉 면적을 COMSOL 접촉 해석으로 얻으면, 강체 겹침 모델과 얼마나 다른가?
16. 실물 기준선(d = 0)을 맞추기 위해 cladding↔공기 경계에 넣을 표면 거칠기나 산란 손실 값은 어떻게 정하는가? (측정값으로 보정 등)
