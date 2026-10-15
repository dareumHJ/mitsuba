## 전체 코드 흐름

make_test_fiber.py ---> stl/*.stl ---> perpare_meshes.py ---> prepared/*.ply ---> trace_mitsuba.py ---> Pout/Pin
(형상 생성; COMSOL 데이터 없을 때)                              (mesh 정리/분류)                              (GPU 광선 추적)

trace_mitsuba.py ---> view_rays.py (3D vis) / check_straight_2d.py (verification)


## Example (COMSOL export 대체)

### 1. Build a shape
```
python make_test_fiber.py --out my_test/stl \
--bend-radius 0.5 \
--length 20

```

### 2. Organize meshes (stl to ply)
```
python prepare_meshes.py \
--core      my_test/stl/core_boundary.stl \
--cladding  my_test/stl/cladding_outer.stl \
--detector  my_test/stl/detector.stl \
--out       my_test/prepared
```

### 3. Run ray traces
```
python trace_mitsuba.py --config my_test/config.yaml --rays 1000000
```

### 4. 3D visualization
```
python view_rays.py --config my_test/config.yaml
```

### 5. (Optional) validation of a result
```
python check_straight_2d.py
```