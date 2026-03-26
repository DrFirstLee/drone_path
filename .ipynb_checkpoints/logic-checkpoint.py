


@app.post("/drone_analyze")
async def drone_analyze(file: UploadFile = File(...)):
    print(f"Drone Analysis Request: {file.filename}")
    
    # 임시 디렉토리 생성하여 처리
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # 파일 저장
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
            
        target_file = temp_file_path
        base_dir = temp_dir
        
        print(f"Reading file: {target_file}")
        
        column_names = [
            'Time', 'Longitude', 'Latitude', 'Altitude', 
            'Roll', 'Pitch', 'Yaw', 
            'Mag1_X', 'Mag1_Y', 'Mag1_Z', 
            'Mag2_X', 'Mag2_Y', 'Mag2_Z'
        ]
        

        try:
            df = pd.read_csv(target_file, header=None)
            
            # Timestamp 컬럼 찾기 및 데이터 시프트 처리
            valid_start_idx = -1
            
            # 처음 5개 컬럼 정도만 검사 (너무 뒤에 있지는 않을 것이라 가정)
            check_limit = min(df.shape[1], 5)
            
            for i in range(check_limit):
                sample_val = df.iloc[0, i]
                numeric_val = float(sample_val)
                if numeric_val <= 500000:
                    continue
                else:
                    valid_start_idx = i
                    print(f"Timestamp column found at index {i}")
                    break

            if valid_start_idx == -1:
                raise ValueError("Timestamp column not found")

            
            if valid_start_idx != -1:
                # 유효한 시작 인덱스 발견 시, 해당 인덱스부터 13개 컬럼 사용
                end_idx = valid_start_idx + len(column_names)
                if df.shape[1] >= end_idx:
                    df = df.iloc[:, valid_start_idx:end_idx]
                    df.columns = column_names
                else:
                    # 컬럼이 부족한 경우 가능한 만큼만 가져옴
                    df = df.iloc[:, valid_start_idx:]
                    df.columns = column_names[:df.shape[1]]
            else:
                print("Could not find a valid timestamp column. Defaulting to column 0.")
                # 기존 로직 (컬럼 0부터 시작)
                if df.shape[1] >= len(column_names):
                    current_cols = column_names + [f"Extra_{i}" for i in range(len(column_names), df.shape[1])]
                    df.columns = current_cols
                else:
                    df.columns = column_names[:df.shape[1]]
                
            print("Data loaded. Applying segmentation...")

            # 3. Feature Engineering
            # Yaw Unwrapping
            yaw_rad = np.deg2rad(df['Yaw'])
            df['Yaw_unwrap'] = np.rad2deg(np.unwrap(yaw_rad))
            
            # Improved Feature Engineering for robust Turn Detection
            # Improved Feature Engineering for robust Turn Detection
            window_size = 30
            
            # 1. IMU-based Features
            # Yaw (Heading)
            df['Yaw_diff'] = df['Yaw_unwrap'].diff().fillna(0).abs()
            df['Yaw_rate'] = df['Yaw_diff'].rolling(window=window_size, center=True).mean().fillna(0)
            df['Yaw_std'] = df['Yaw_unwrap'].rolling(window=window_size, center=True).std().fillna(0)
            
            # Roll (Banking)
            df['Roll_abs'] = df['Roll'].abs().rolling(window=window_size, center=True).mean().fillna(0)
            
            # 2. Centripetal Acceleration (v * omega)
            # This helps distinguish 'High Speed Turns' (High Centripetal) vs 'Straight' (Low Centripetal)
            # and 'Stationary Turns' (Low Centripetal, but High Yaw Rate).
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                R = 6371000 # Earth radius
                lat_rad = np.deg2rad(df['Latitude'])
                lon_rad = np.deg2rad(df['Longitude'])
                
                # Approximate distance per sample (meter)
                # dy = R * dLat
                # dx = R * dLon * cos(Lat)
                dx = R * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
                dy = R * lat_rad.diff().fillna(0)
                ds = np.sqrt(dx**2 + dy**2)
                
                # Speed proxy (ds per sample)
                df['Speed_proxy'] = ds.rolling(window=window_size, center=True).mean().fillna(0)
                
                # Centripetal Acceleration ~ Speed * YawRate
                # Note: Yaw_rate is deg/sample, Speed is m/sample. Product is proportional to m/s^2.
                df['Centripetal_Acc'] = df['Speed_proxy'] * df['Yaw_rate']
            else:
                df['Speed_proxy'] = 0
                df['Centripetal_Acc'] = 0
            
            # Combine features
            # Yaw_rate: Detects sharp/stationary turns
            # Centripetal_Acc: Detects high-speed/wide turns
            # Roll_abs: Auxiliary indicator for banked turns
            feature_cols = ['Yaw_rate', 'Yaw_std', 'Roll_abs', 'Centripetal_Acc']
            df[feature_cols] = df[feature_cols].fillna(0)
            
            # Scaling & Clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[feature_cols])
            
            # HMM (Hidden Markov Model) for Time-Series Continuity
            # Ideally better than K-Means for sequential data as it models transitions.
            from hmmlearn import hmm
            print("Using GaussianHMM for segmentation...")
            
            # 2 states: Straight vs Turn
            # covariance_type="diag" is generally robust for this
            model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(X_scaled)
            
            # Predict states
            df['State_Raw'] = model.predict(X_scaled)
            
            # Identify which state is 'Turn'
            # Check the mean of features for each state. The state with higher values is likely 'Turn'.
            state_means = []
            for s in range(2):
                # Average feature value for this state
                state_mean = np.mean(X_scaled[df['State_Raw'] == s], axis=0)
                state_means.append(np.sum(state_mean)) # Sum of standardized features
            
            turn_label = np.argmax(state_means)

            # try:
            #     from hmmlearn import hmm
            #     print("Using GaussianHMM for segmentation...")
                
            #     # 2 states: Straight vs Turn
            #     # covariance_type="diag" is generally robust for this
            #     model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            #     model.fit(X_scaled)
                
            #     # Predict states
            #     df['State_Raw'] = model.predict(X_scaled)
                
            #     # Identify which state is 'Turn'
            #     # Check the mean of features for each state. The state with higher values is likely 'Turn'.
            #     state_means = []
            #     for s in range(2):
            #         # Average feature value for this state
            #         state_mean = np.mean(X_scaled[df['State_Raw'] == s], axis=0)
            #         state_means.append(np.sum(state_mean)) # Sum of standardized features
                
            #     turn_label = np.argmax(state_means)
                
            # except ImportError:
            #     print("hmmlearn not found, falling back to K-Means...")
            #     # Fallback to K-Means if hmmlearn is missing
            #     kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            #     df['State_Raw'] = kmeans.fit_predict(X_scaled)
                
            #     cluster_centers = kmeans.cluster_centers_
            #     cluster_magnitudes = np.sum(cluster_centers, axis=1)
            #     turn_label = np.argmax(cluster_magnitudes)
            
            # 0: Straight, 1: Turn 으로 라벨 정규화
            df['State'] = df['State_Raw'].apply(lambda x: 1 if x == turn_label else 0)
            
            # [DEBUG] RAW Clustering Visualization Mode
            # Bypassing complex post-processing to see raw HMM/KMeans output.
            
            # 4. Smoothing (노이즈 제거) - Skipped
            # smooth_window = 10
            # df['State_Smooth'] = df['State'].rolling(window=smooth_window, center=True).mean()
            # df['State_Smooth'] = df['State_Smooth'].apply(lambda x: 1 if x > 0.5 else 0).fillna(0).astype(int)
            df['State_Smooth'] = df['State'] # Use Raw State
            
            # 5. Segment Identification & Merging (구간 나누기 및 병합) - Simplified
            df['Segment_Change'] = df['State_Smooth'].diff().fillna(0).abs()
            # Simple cumulative sum to create unique IDs for every state change
            df['Final_Segment_ID'] = df['Segment_Change'].cumsum().astype(int)

            # 6. Rotate_Error 판별: PCA 선형성 + Yaw 총 변화량 복합 판단
            #
            #  [문제] 130° 꺾임(헤어핀/V자) 경로는 두 팔이 반대 방향이라
            #         PCA 1st 성분이 높게 나와 직선으로 오판될 수 있음.
            #
            #  [해결] GPS 선형성이 높아도 Yaw 총 변화량이 크면 실제 회전으로 인정.
            #         → 두 조건이 모두 성립해야 Rotate_Error:
            #           (1) PCA 선형성 높음 (GPS 경로가 직선형)
            #           (2) Yaw 총 변화 작음 (방향이 거의 안 바뀜)
            #
            # State_Final: 0=Straight, 1=Rotate, 3=Rotate_Error
            from sklearn.decomposition import PCA as _PCA

            def is_linear_path(lats, lons, linearity_threshold=0.98):
                """GPS 좌표 집합에 PCA를 적용하여 경로의 선형성을 측정.

                반환값: (is_linear (bool), pca_ratio (float))
                """
                if len(lats) < 6:
                    return False, 0.0

                R = 6371000.0
                lat_rad = np.deg2rad(np.mean(lats))
                xs = np.deg2rad(lons) * R * np.cos(lat_rad)
                ys = np.deg2rad(lats) * R
                coords = np.column_stack([xs, ys])

                span = np.ptp(coords, axis=0)
                if np.max(span) < 1.0:  # 이동 반경 1m 미만 → 제자리
                    return True, 1.0

                pca = _PCA(n_components=2)
                pca.fit(coords)
                ratio = pca.explained_variance_ratio_[0]
                return ratio >= linearity_threshold, ratio

            df['State_Final'] = df['State_Smooth'].copy()  # 기본값 복사 (0=Straight)
            df['PCA_Ratio'] = 0.0

            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                rotate_seg_ids = df[df['State_Smooth'] == 1]['Final_Segment_ID'].unique()
                for rid in rotate_seg_ids:
                    seg_mask = (df['Final_Segment_ID'] == rid)
                    seg = df[seg_mask]
                    lats = seg['Latitude'].values
                    lons = seg['Longitude'].values

                    gps_is_linear, pca_ratio = is_linear_path(lats, lons)
                    df.loc[seg_mask, 'PCA_Ratio'] = pca_ratio

                    # Yaw 총 변화량: 해당 구간의 unwrapped yaw 범위
                    # - 직선 비행: yaw 거의 안 변함 → 작은 값
                    # - 90°/130° 실제 턴: yaw 크게 변함 → 큰 값
                    if 'Yaw_unwrap' in seg.columns and len(seg) > 0:
                        yaw_change = seg['Yaw_unwrap'].max() - seg['Yaw_unwrap'].min()
                    else:
                        yaw_change = 0.0

                    # [판정]
                    # 1. 원본 로직 적용 (기본) 
                    YAW_CHANGE_THRESHOLD = 45.0  # 도(°).
                    
                    # 2. 먼저 Rotate_Error 여부를 판별합니다. (Yaw가 거의 변하지 않고 GPS 궤적도 직선일 때)
                    if gps_is_linear and yaw_change < YAW_CHANGE_THRESHOLD:
                        df.loc[seg_mask, 'State_Final'] = 3  # Rotate_Error (오분류)
                    else:
                        if gps_is_linear:
                            df.loc[seg_mask, 'State_Final'] = 2  # Rotate_Line (직선처럼 보이는 회전)
                        else:
                            df.loc[seg_mask, 'State_Final'] = 1  # 정상 Rotate
            else:
                df.loc[df['State_Smooth'] == 1, 'State_Final'] = 1
                
            # --- 세그먼트 ID 재계산 ---
            # 서브 클러스터링으로 인해 State_Final이 쪼개졌으므로, Final_Segment_ID를 새로 부여합니다.
            df['Segment_Change_Final'] = (df['State_Final'] != df['State_Final'].shift(1)).astype(int)
            df['Final_Segment_ID'] = df['Segment_Change_Final'].cumsum().astype(int)


            # 7. Line ↔ Rotate_Error/Rotate_Line 체인 병합
            #    실제 Rotate(1)로 끊기지 않는 한, {Line(0), Rotate_Line(2), Rotate_Error(3)} 인접 구간들을
            #    하나의 "run"으로 묶어서 병합.
            #
            #    병합 규칙:
            #      - run 내(1로 끊기기 전까지의 구간)에 Line(0)이 하나라도 존재한다면,
            #        그 run 안에 있는 모든 2와 3은 사실상 Line의 연장선이므로 모두 Line(0)으로 통합한다.
            #      - 만약 run 안에 Line(0)이 아예 없다면 (예: 부분적으로 2나 3만 있는 경우) 그대로 둔다.

            # 세그먼트 순서 추출 (원본 인덱스 기반 시간 순 정렬)
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
            ))  # [(seg_id, state_final), ...]

            # 실제 Rotate(1) 경계로 run 분리
            runs = []
            current_run = []
            for seg_id, state in ordered_segs:
                if state in (0, 2, 3):
                    current_run.append((seg_id, state))
                else:  # state == 1: 실제 Rotate → 현재 run 종료
                    if current_run:
                        runs.append(current_run)
                        current_run = []
                    runs.append([(seg_id, state)])  # Rotate 자체는 단독 run
            if current_run:
                runs.append(current_run)

            # 각 run 내에서 Line(0)이 하나라도 있다면, 
            # 인접한 Rotate_Line(2)와 Rotate_Error(3)를 모두 해당 Line으로 병합합니다.
            # (Sub-clustering으로 인해 1-2-0 처럼 잘린 경우에도 2가 0으로 병합되도록 함)
            merge_ops = {}  # {old_seg_id: new_seg_id}
            for run in runs:
                # Rotate(1) 단독 run이면 스킵
                if len(run) == 1 and run[0][1] == 1:
                    continue

                line_positions = [i for i, (_, st) in enumerate(run) if st == 0]

                # Run 내에 Line(0)이 아예 없으면 병합 대상 없음 (예: [2], [3], [2, 3] 등 서로들만 있는 경우)
                if len(line_positions) == 0:
                    continue

                # Run 내에 Line(0)이 하나라도 있다면, Run 전체 요소를 가장 첫번째 Line의 ID로 병합
                target_id = run[line_positions[0]][0]

                for old_id, _ in run:
                    if old_id != target_id:
                        merge_ops[old_id] = target_id

            # 병합 적용
            n_merged = 0
            all_target_ids = set(merge_ops.values())
            for old_id, new_id in merge_ops.items():
                df.loc[df['Final_Segment_ID'] == old_id, 'Final_Segment_ID'] = new_id
                n_merged += 1

            # 병합된 그룹 내 남은 Error/Line 판단 구간들을 Line(0)으로 교정
            for tid in all_target_ids:
                df.loc[(df['Final_Segment_ID'] == tid) & (df['State_Final'].isin([2, 3])), 'State_Final'] = 0

            if n_merged > 0:
                print(f"[Merge] {n_merged}개 세그먼트를 Line으로 병합했습니다.")
                
            # 병합 이후 중간에 건너뛴(이빨 빠진) 세그먼트 ID들을 1번부터 차례대로 재정렬합니다.
            alive_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
            id_mapping = {old_id: new_idx for new_idx, old_id in enumerate(alive_seg_ids, start=1)}
            df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'] = df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'].map(id_mapping).astype(int)
            
            # 7. 지도 시각화
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                center_lat = df['Latitude'].mean()
                center_lon = df['Longitude'].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
                
                # 전체 경로를 연한 회색으로 표시 (배경)
                all_coords = df[['Latitude', 'Longitude']].values.tolist()
                folium.PolyLine(
                    locations=all_coords,
                    color="#000000", 
                    weight=5,
                    opacity=0.5,
                    tooltip="Original Path"
                ).add_to(m)
                
                # 다양한 색상 팔레트 (검정색 배경과 확실히 구분되는 밝고 선명한 색상 위주)
                colors = [
                    '#FF0000', # Red
                    '#00FF00', # Lime
                    '#0000FF', # Blue
                    '#FFFF00', # Yellow
                    '#FF00FF', # Magenta
                    '#00FFFF', # Cyan
                    '#FF8000', # Orange
                    '#0080FF', # Azure
                    '#FF0080', # Deep Pink
                    '#80FF00', # Chartreuse
                    '#00FF80', # Spring Green
                    '#FF4040', # Light Red
                    '#4040FF', # Light Blue
                    '#FFD700', # Gold
                    '#ADFF2F', # Green Yellow
                    '#FF69B4', # Hot Pink
                    '#1E90FF', # Dodger Blue
                    '#DC143C', # Crimson
                ]
                
                sorted_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
                
                for seg_id in sorted_seg_ids:
                    group = df[df['Final_Segment_ID'] == seg_id]
                    coords = group[['Latitude', 'Longitude']].values.tolist()

                    # State_Final 기준으로 세그먼트 타입 결정
                    # 0=Straight, 1=Rotate, 2=Rotate_Line, 3=Rotate_Error
                    seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]

                    if seg_state_final == 0:
                        type_str = "Straight"
                        label_prefix = "Line"
                        color_idx = seg_id % len(colors)
                        color = colors[color_idx]
                        weight = 3
                        opacity = 0.8
                        dash_array = None
                    elif seg_state_final == 1:
                        type_str = "Rotate"
                        label_prefix = "Rotate"
                        color_idx = seg_id % len(colors)
                        color = colors[color_idx]
                        weight = 5
                        opacity = 0.9
                        dash_array = '5, 5'
                    elif seg_state_final == 2:
                        type_str = "Rotate_Line"
                        label_prefix = "Rotate_Line"
                        color = '#9400D3'  # 눈에 띄는 보라색으로 설정
                        weight = 5
                        opacity = 0.9
                        dash_array = '10, 5'  # 점선으로 구분
                    else:  # state == 3: Rotate_Error
                        type_str = "Rotate_Error"
                        label_prefix = "Rotate_Error"
                        color = '#FF6600'  # 눈에 띄는 주황색으로 고정
                        weight = 5
                        opacity = 0.9
                        dash_array = '10, 5'  # 더 긴 점선으로 구분

                    label_text = f"{label_prefix}-{seg_id}"
                    
                    if (seg_state_final != 0 )and 'PCA_Ratio' in group.columns:
                        pca_val = group['PCA_Ratio'].iloc[0]
                        label_text += f" (PCA: {pca_val:.3f})"

                    # --- [FIX] 시간 불연속 구간 감지 및 분리하여 그리기 ---
                    # "동그란 부분이 왜 직선표현되지" -> 떨어져 있는 같은 ID 구간을 이어서 그려서 발생하는 문제 해결

                    # 그룹 내에서 원래 인덱스를 가져와서 차이 계산
                    indices = group.index.values
                    if len(indices) > 0:
                        # 인덱스가 1보다 큰 차이가 나면 불연속으로 간주
                        idx_diff = np.diff(indices)
                        split_locs = np.where(idx_diff > 1)[0] + 1

                        # 서브 그룹으로 분할
                        sub_groups_indices = np.split(indices, split_locs)

                        for sub_indices in sub_groups_indices:
                            if len(sub_indices) < 2: continue  # 점 1개는 선으로 못 그림

                            sub_group = df.loc[sub_indices]
                            sub_coords = sub_group[['Latitude', 'Longitude']].values.tolist()

                            folium.PolyLine(
                                locations=sub_coords,
                                color=color,
                                weight=weight,
                                opacity=opacity,
                                dash_array=dash_array,
                                tooltip=f"ID {seg_id}: {type_str}"
                            ).add_to(m)

                            # 각 서브 그룹의 시작점에 작은 마커 표시
                            folium.CircleMarker(
                                location=sub_coords[0],
                                radius=3 if seg_state_final == 3 else 2,
                                color=color,
                                fill=True,
                                popup=f"ID {seg_id} [{type_str}] part start"
                            ).add_to(m)

                    center_lat_g = group['Latitude'].mean()
                    center_lon_g = group['Longitude'].mean()

                    # Rotate_Error는 텍스트 색상 강조 + ⚠️ 아이콘 붙여서 구분
                    if seg_state_final == 2:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 2px white;">🔄 {label_text}</div>'
                    elif seg_state_final == 3:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: #FF6600; text-shadow: 1px 1px 2px white;">⚠️ {label_text}</div>'
                    else:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 1px white;">{label_text}</div>'

                    folium.Marker(
                        location=[center_lat_g, center_lon_g],
                        icon=folium.DivIcon(
                            icon_size=(180, 36),
                            icon_anchor=(0, 0),
                            html=label_html
                        )
                    ).add_to(m)

                output_file = os.path.join(temp_dir, "flight_path_segmented.html")
                m.save(output_file)
                print(f"지도 파일 저장 완료: {output_file}")
                
                # 7. 분할된 데이터 저장 (./splited) 및 Download Link 생성
                output_split_dir = os.path.join(temp_dir, "splited")
                if not os.path.exists(output_split_dir):
                    os.makedirs(output_split_dir)
                    
                csv_links = []
                
                # Sort group keys for better UI ordering
                sorted_groups = sorted(df[df['Final_Segment_ID'] != -1].groupby('Final_Segment_ID'), key=lambda x: x[0])
                
                for seg_id, group in sorted_groups:
                    # State_Final 기준으로 레이블 결정: 0=Line, 1=Rotate, 2=Rotate_Line, 3=Rotate_Error
                    seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]
                    if seg_state_final == 0:
                        label_prefix = "Line"
                        link_style = "display:block; margin:5px 0; color:#333;"
                    elif seg_state_final == 1:
                        label_prefix = "Rotate"
                        link_style = "display:block; margin:5px 0; color:#0055aa; font-weight:bold;"
                    elif seg_state_final == 2:
                        label_prefix = "Rotate_Line"
                        link_style = "display:block; margin:5px 0; color:#9400D3; font-weight:bold;"
                    else:  # 3 = Rotate_Error
                        label_prefix = "Rotate_Error"
                        link_style = "display:block; margin:5px 0; color:#FF6600; font-weight:bold;"

                    file_name = f"{label_prefix}-{seg_id}.csv"
                    save_path = os.path.join(output_split_dir, file_name)

                    # Generate CSV content
                    csv_content = group.to_csv(index=True, header=True)

                    # Save for record
                    with open(save_path, "w", encoding='utf-8') as f:
                        f.write(csv_content)

                    # Create Data URI for download
                    b64_csv = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
                    prefix_icon = "⚠️ " if seg_state_final == 3 else ("🔄 " if seg_state_final == 2 else "")
                    download_link = f'<a href="data:text/csv;base64,{b64_csv}" download="{file_name}" style="{link_style}">{prefix_icon}Download {file_name}</a>'
                    csv_links.append(download_link)
                
                # HTML 파일 읽기 (반환용)
                with open(output_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # HTML에 Download Panel 추가
                if csv_links:
                    download_panel = f"""
                    <div style="position: fixed; top: 10px; right: 10px; width: 250px; max-height: 80vh; overflow-y: auto; 
                                background-color: white; padding: 10px; border: 2px solid #ccc; z-index: 9999; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
                        <h4>Segment CSV Downloads</h4>
                        {''.join(csv_links)}
                    </div>
                    """
                    # Insert before </body>
                    if "</body>" in html_content:
                        html_content = html_content.replace("</body>", f"{download_panel}</body>")
                    else:
                        html_content += download_panel

                return HTMLResponse(content=html_content)
                
        except Exception as e:
            print(f"오류 발생: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

