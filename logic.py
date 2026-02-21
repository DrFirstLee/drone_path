"""
드론 비행 경로 분석 스크립트
==============================
사용법:
    python logic.py                     # input.csv → .output/
    python logic.py --input my.csv      # 파일 직접 지정
    python logic.py --input my.csv --output results/

출력물:
    .output/
    ├── flight_path_segmented.html   ← Folium 지도
    ├── full_result.csv              ← 전체 분석 결과 (State_Final 컬럼 포함)
    ├── Line-0.csv
    ├── Line-2.csv
    ├── Rotate-1.csv
    ├── Rotate_Error-3.csv
    └── ...
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as _PCA

# ──────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="드론 비행 경로 분석")
    parser.add_argument("--input",  default="input.csv",  help="입력 CSV 파일 경로 (기본값: input.csv)")
    parser.add_argument("--output", default=".output",    help="결과 저장 폴더 (기본값: .output)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Rotate_Error 판별용 PCA 선형성 검사
# ──────────────────────────────────────────────
def is_linear_path(lats, lons, linearity_threshold=0.97):
    """GPS 좌표 집합에 PCA를 적용하여 경로의 선형성을 측정.

    반환값:
        True  → 직선에 가까운 GPS 형태
        False → 2차원으로 퍼진 GPS 형태 (원/8자)
    """
    if len(lats) < 6:
        return False

    R = 6371000.0
    lat_rad = np.deg2rad(np.mean(lats))
    xs = np.deg2rad(lons) * R * np.cos(lat_rad)
    ys = np.deg2rad(lats) * R
    coords = np.column_stack([xs, ys])

    span = np.ptp(coords, axis=0)
    if np.max(span) < 1.0:   # 이동 반경 1m 미만 → 제자리
        return True

    pca = _PCA(n_components=2)
    pca.fit(coords)
    return pca.explained_variance_ratio_[0] >= linearity_threshold


# ──────────────────────────────────────────────
# 핵심 분석 함수
# ──────────────────────────────────────────────
def analyze(input_path: str, output_dir: str):
    print(f"\n{'='*55}")
    print(f"  드론 비행 경로 분석 시작")
    print(f"  입력: {input_path}")
    print(f"  출력: {output_dir}")
    print(f"{'='*55}\n")

    # ── 출력 폴더 생성 ──────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. CSV 읽기 ─────────────────────────────
    column_names = [
        'Time', 'Longitude', 'Latitude', 'Altitude',
        'Roll', 'Pitch', 'Yaw',
        'Mag1_X', 'Mag1_Y', 'Mag1_Z',
        'Mag2_X', 'Mag2_Y', 'Mag2_Z'
    ]

    df = pd.read_csv(input_path, header=None)
    print(f"[1/7] CSV 로드 완료: {len(df)} 행, {df.shape[1]} 열")

    # Timestamp 컬럼 찾기 (값 > 500000 인 첫 컬럼)
    valid_start_idx = -1
    check_limit = min(df.shape[1], 5)
    for i in range(check_limit):
        try:
            val = float(df.iloc[0, i])
            if val > 500000:
                valid_start_idx = i
                print(f"    Timestamp 컬럼 위치: 인덱스 {i}")
                break
        except (ValueError, TypeError):
            continue

    if valid_start_idx == -1:
        raise ValueError("Timestamp 컬럼을 찾을 수 없습니다. 첫 5개 컬럼에 Unix timestamp(> 500000)가 없습니다.")

    end_idx = valid_start_idx + len(column_names)
    if df.shape[1] >= end_idx:
        df = df.iloc[:, valid_start_idx:end_idx].copy()
    else:
        df = df.iloc[:, valid_start_idx:].copy()
    df.columns = column_names[:df.shape[1]]

    # ── 2. Feature Engineering ──────────────────
    print("[2/7] 피처 엔지니어링...")

    yaw_rad = np.deg2rad(df['Yaw'])
    df['Yaw_unwrap'] = np.rad2deg(np.unwrap(yaw_rad))

    window_size = 30
    df['Yaw_diff']  = df['Yaw_unwrap'].diff().fillna(0).abs()
    df['Yaw_rate']  = df['Yaw_diff'].rolling(window=window_size, center=True).mean().fillna(0)
    df['Yaw_std']   = df['Yaw_unwrap'].rolling(window=window_size, center=True).std().fillna(0)
    df['Roll_abs']  = df['Roll'].abs().rolling(window=window_size, center=True).mean().fillna(0)

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        R = 6371000
        lat_rad = np.deg2rad(df['Latitude'])
        lon_rad = np.deg2rad(df['Longitude'])
        dx = R * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
        dy = R * lat_rad.diff().fillna(0)
        ds = np.sqrt(dx**2 + dy**2)
        df['Speed_proxy']    = ds.rolling(window=window_size, center=True).mean().fillna(0)
        df['Centripetal_Acc'] = df['Speed_proxy'] * df['Yaw_rate']
    else:
        df['Speed_proxy']    = 0
        df['Centripetal_Acc'] = 0

    feature_cols = ['Yaw_rate', 'Yaw_std', 'Roll_abs', 'Centripetal_Acc']
    df[feature_cols] = df[feature_cols].fillna(0)

    # ── 3. HMM 클러스터링 ───────────────────────
    print("[3/7] HMM 클러스터링 (직선 vs 회전)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    try:
        from hmmlearn import hmm
        print("    GaussianHMM 사용 중...")
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_scaled)
        df['State_Raw'] = model.predict(X_scaled)
        state_means = []
        for s in range(2):
            state_mean = np.mean(X_scaled[df['State_Raw'] == s], axis=0)
            state_means.append(np.sum(state_mean))
        turn_label = int(np.argmax(state_means))
    except ImportError:
        print("    hmmlearn 없음 → K-Means 사용...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        df['State_Raw'] = kmeans.fit_predict(X_scaled)
        cluster_magnitudes = np.sum(kmeans.cluster_centers_, axis=1)
        turn_label = int(np.argmax(cluster_magnitudes))

    df['State'] = (df['State_Raw'] == turn_label).astype(int)
    df['State_Smooth'] = df['State']   # 스무딩 미적용 (Raw 사용)

    # ── 4. 세그먼트 ID 부여 ─────────────────────
    print("[4/7] 세그먼트 ID 부여...")

    df['Segment_Change']   = df['State_Smooth'].diff().fillna(0).abs()
    df['Final_Segment_ID'] = df['Segment_Change'].cumsum().astype(int)

    # ── 5. Rotate_Error 판별 ────────────────────
    #   조건 AND: GPS 선형성(PCA ≥ 0.97) AND Yaw 변화 < 45°
    print("[5/7] Rotate_Error 판별 (PCA 선형성 + Yaw 변화량)...")

    YAW_CHANGE_THRESHOLD = 45.0   # 도(°)
    df['State_Final'] = df['State_Smooth'].copy()

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        rotate_seg_ids = df[df['State_Smooth'] == 1]['Final_Segment_ID'].unique()
        for rid in rotate_seg_ids:
            seg_mask = df['Final_Segment_ID'] == rid
            seg      = df[seg_mask]
            lats     = seg['Latitude'].values
            lons     = seg['Longitude'].values

            gps_is_linear = is_linear_path(lats, lons)
            yaw_change    = seg['Yaw_unwrap'].max() - seg['Yaw_unwrap'].min() if len(seg) > 0 else 0.0

            if gps_is_linear and yaw_change < YAW_CHANGE_THRESHOLD:
                df.loc[seg_mask, 'State_Final'] = 3   # Rotate_Error
            else:
                df.loc[seg_mask, 'State_Final'] = 1   # 정상 Rotate
    else:
        df.loc[df['State_Smooth'] == 1, 'State_Final'] = 1

    n_line  = (df['State_Final'] == 0).sum()
    n_rot   = (df['State_Final'] == 1).sum()
    n_err   = (df['State_Final'] == 3).sum()
    print(f"    Line: {n_line}행  Rotate: {n_rot}행  Rotate_Error: {n_err}행")

    # ── 6. Line ↔ Rotate_Error 체인 병합 ────────
    print("[6/7] Line-Error 체인 병합...")

    seg_order_df = (
        df[df['Final_Segment_ID'] >= 0]
        .groupby('Final_Segment_ID')
        .agg(
            state_final=('State_Final', 'first'),
            first_idx=('Final_Segment_ID', lambda x: x.index.min())
        )
        .reset_index()
        .sort_values('first_idx')
    )
    ordered_segs = list(zip(
        seg_order_df['Final_Segment_ID'].tolist(),
        seg_order_df['state_final'].tolist()
    ))

    # 실제 Rotate(1) 경계로 run 분리
    runs, current_run = [], []
    for seg_id, state in ordered_segs:
        if state in (0, 3):
            current_run.append((seg_id, state))
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
            runs.append([(seg_id, state)])
    if current_run:
        runs.append(current_run)

    # 각 run 내 첫 Line ~ 마지막 Line 사이 병합
    merge_ops = {}
    for run in runs:
        if len(run) == 1 and run[0][1] == 1:
            continue
        line_positions = [i for i, (_, st) in enumerate(run) if st == 0]
        error_count   = sum(1 for _, st in run if st == 3)
        if len(line_positions) < 2 or error_count == 0:
            continue
        first_line_pos = line_positions[0]
        last_line_pos  = line_positions[-1]
        target_id      = run[first_line_pos][0]
        for i in range(first_line_pos + 1, last_line_pos + 1):
            old_id, _ = run[i]
            if old_id != target_id:
                merge_ops[old_id] = target_id

    all_target_ids = set(merge_ops.values())
    for old_id, new_id in merge_ops.items():
        df.loc[df['Final_Segment_ID'] == old_id, 'Final_Segment_ID'] = new_id
    for tid in all_target_ids:
        df.loc[(df['Final_Segment_ID'] == tid) & (df['State_Final'] == 3), 'State_Final'] = 0

    if merge_ops:
        print(f"    {len(merge_ops)}개 세그먼트를 Line으로 병합 완료")

    # ── 7. 지도 시각화 + CSV 저장 ───────────────
    print("[7/7] 지도 시각화 및 결과 저장...")

    colors = [
        '#FF0000', '#00CC00', '#0000FF', '#FFD700', '#FF00FF',
        '#00FFFF', '#FF8000', '#0080FF', '#FF0080', '#80FF00',
        '#00FF80', '#FF4040', '#4040FF', '#ADFF2F', '#FF69B4',
        '#1E90FF', '#DC143C', '#8B00FF',
    ]

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

    # 배경 전체 경로 (검정 반투명)
    all_coords = df[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(
        locations=all_coords,
        color="#000000", weight=5, opacity=0.4,
        tooltip="Original Path"
    ).add_to(m)

    sorted_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())

    for seg_id in sorted_seg_ids:
        group           = df[df['Final_Segment_ID'] == seg_id]
        seg_state_final = group['State_Final'].iloc[0]

        if seg_state_final == 0:
            type_str     = "Straight"
            label_prefix = "Line"
            color        = colors[seg_id % len(colors)]
            weight, opacity, dash_array = 3, 0.85, None
        elif seg_state_final == 1:
            type_str     = "Rotate"
            label_prefix = "Rotate"
            color        = colors[seg_id % len(colors)]
            weight, opacity, dash_array = 5, 0.9, '5, 5'
        else:   # 3 = Rotate_Error
            type_str     = "Rotate_Error"
            label_prefix = "Rotate_Error"
            color        = '#FF6600'
            weight, opacity, dash_array = 5, 0.9, '10, 5'

        label_text = f"{label_prefix}-{seg_id}"

        # 시간 불연속 구간 분리 후 별도 선 그리기 (동그란 구간 직선화 버그 방지)
        indices    = group.index.values
        idx_diff   = np.diff(indices)
        split_locs = np.where(idx_diff > 1)[0] + 1
        sub_groups_indices = np.split(indices, split_locs)

        for sub_indices in sub_groups_indices:
            if len(sub_indices) < 2:
                continue
            sub_group  = df.loc[sub_indices]
            sub_coords = sub_group[['Latitude', 'Longitude']].values.tolist()

            folium.PolyLine(
                locations=sub_coords,
                color=color, weight=weight, opacity=opacity,
                dash_array=dash_array,
                tooltip=f"ID {seg_id}: {type_str}"
            ).add_to(m)

            folium.CircleMarker(
                location=sub_coords[0],
                radius=3 if seg_state_final == 3 else 2,
                color=color, fill=True,
                popup=f"ID {seg_id} [{type_str}] 시작점"
            ).add_to(m)

        # 중앙 레이블 마커
        center_lat_g = group['Latitude'].mean()
        center_lon_g = group['Longitude'].mean()
        if seg_state_final == 3:
            label_html = (
                f'<div style="font-size:10pt; font-weight:bold; color:#FF6600; '
                f'text-shadow:1px 1px 2px white;">⚠️ {label_text}</div>'
            )
        else:
            label_html = (
                f'<div style="font-size:10pt; font-weight:bold; color:{color}; '
                f'text-shadow:1px 1px 1px white;">{label_text}</div>'
            )
        folium.Marker(
            location=[center_lat_g, center_lon_g],
            icon=folium.DivIcon(icon_size=(180, 36), icon_anchor=(0, 0), html=label_html)
        ).add_to(m)

    # ── HTML 저장 ────────────────────────────────
    html_path = os.path.join(output_dir, "flight_path_segmented.html")
    m.save(html_path)
    print(f"    지도 저장: {html_path}")

    # ── 전체 결과 CSV 저장 ───────────────────────
    full_csv_path = os.path.join(output_dir, "full_result.csv")
    df.to_csv(full_csv_path, index=True, encoding="utf-8-sig")
    print(f"    전체 결과 CSV 저장: {full_csv_path}")

    # ── 세그먼트별 CSV 저장 ──────────────────────
    sorted_groups = sorted(
        df[df['Final_Segment_ID'] != -1].groupby('Final_Segment_ID'),
        key=lambda x: x[0]
    )
    saved_csvs = []
    for seg_id, group in sorted_groups:
        seg_state_final = group['State_Final'].iloc[0]
        if seg_state_final == 0:
            label_prefix = "Line"
        elif seg_state_final == 1:
            label_prefix = "Rotate"
        else:
            label_prefix = "Rotate_Error"

        file_name = f"{label_prefix}-{seg_id}.csv"
        save_path = os.path.join(output_dir, file_name)
        group.to_csv(save_path, index=True, encoding="utf-8-sig")
        saved_csvs.append(file_name)

    print(f"    세그먼트 CSV {len(saved_csvs)}개 저장 완료")

    # ── 결과 요약 ────────────────────────────────
    print(f"\n{'='*55}")
    print("  분석 완료 ✅")
    print(f"{'='*55}")
    print(f"  출력 폴더  : {os.path.abspath(output_dir)}")
    print(f"  지도 HTML  : flight_path_segmented.html")
    print(f"  전체 CSV   : full_result.csv")
    print(f"  세그먼트 수: {len(sorted_seg_ids)}개")

    line_segs = [sid for sid in sorted_seg_ids if df[df['Final_Segment_ID'] == sid]['State_Final'].iloc[0] == 0]
    rot_segs  = [sid for sid in sorted_seg_ids if df[df['Final_Segment_ID'] == sid]['State_Final'].iloc[0] == 1]
    err_segs  = [sid for sid in sorted_seg_ids if df[df['Final_Segment_ID'] == sid]['State_Final'].iloc[0] == 3]
    print(f"    Line         : {len(line_segs)}개")
    print(f"    Rotate       : {len(rot_segs)}개")
    print(f"    Rotate_Error : {len(err_segs)}개")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    try:
        analyze(input_path=args.input, output_dir=args.output)
    except Exception as e:
        print(f"\n[오류] 분석 중 예외 발생:\n  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)