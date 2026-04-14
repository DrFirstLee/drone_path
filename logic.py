
@app.post("/drone_analyze_xy")
async def drone_analyze_xy(file: UploadFile = File(...)):
    print(f"Drone Analysis XY Request: {file.filename}")
    
    # 임시 디렉토리 생성하여 처리
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # 파일 저장
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
            
        target_file = temp_file_path
        print(f"Reading file: {target_file}")
        
        try:
            df_orig = pd.read_csv(target_file, header=None)
            df = df_orig.copy()
            
            # Timestamp 컬럼 찾기
            valid_start_idx = -1
            check_limit = min(df.shape[1], 5)
            
            for i in range(check_limit):
                try:
                    if float(df.iloc[0, i]) > 500000:
                        valid_start_idx = i
                        print(f"Timestamp column found at index {i}")
                        break
                except ValueError:
                    continue

            # 1. Roll, Pitch, Yaw 버리고 Time, Longitude, Latitude만 추출
            if valid_start_idx != -1 and df.shape[1] >= valid_start_idx + 3:
                df = df.iloc[:, [valid_start_idx, valid_start_idx+1, valid_start_idx+2]]
                df.columns = ['Time', 'Longitude', 'Latitude']
            else:
                print("Could not properly align columns. Defaulting to first 3 columns.")
                df = df.iloc[:, [0, 1, 2]]
                df.columns = ['Time', 'Longitude', 'Latitude']

            # 결측치 및 문자열 에러 방지
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df = df.dropna(subset=['Latitude', 'Longitude'])
            
            print("Data loaded. Applying Lon/Lat only segmentation...")

            # 2. X, Y 좌표만으로 비행 물리량 도출 및 전처리 (Notebook 로직 이식)
            R = 6371000.0

            # GPS 좌표 미세 노이즈 사전 평활화
            smooth_n = 5
            df['Latitude_s'] = df['Latitude'].rolling(smooth_n, center=True).mean().bfill().ffill()
            df['Longitude_s'] = df['Longitude'].rolling(smooth_n, center=True).mean().bfill().ffill()

            lat_rad = np.deg2rad(df['Latitude_s'])
            lon_rad = np.deg2rad(df['Longitude_s'])

            dx = R * lon_rad.diff().fillna(0) * np.cos(lat_rad.mean())
            dy = R * lat_rad.diff().fillna(0)

            # 드론이 코너에서 멈칫할 때 발생하는 방향(Heading) 노이즈 제거
            ds = np.sqrt(dx**2 + dy**2)
            heading_rad = np.where(ds > 0.1, np.arctan2(dy, dx), np.nan)
            df['Pseudo_Yaw'] = np.rad2deg(heading_rad)
            df['Pseudo_Yaw'] = df['Pseudo_Yaw'].ffill().bfill() 

            yaw_rad_unwrapped = np.unwrap(np.deg2rad(df['Pseudo_Yaw']))
            df['Yaw_unwrap'] = np.rad2deg(yaw_rad_unwrapped)

            # 윈도우 사이즈 축소 (짧고 급격한 코너 포착)
            window_size = 15
            df['Yaw_diff'] = df['Yaw_unwrap'].diff().fillna(0).abs()
            df['Yaw_rate'] = df['Yaw_diff'].rolling(window=window_size, center=True).mean().fillna(0)

            # 핵심 피처: 구간 내 최대 각도 변화량 (Max - Min)
            df['Yaw_range'] = (df['Yaw_unwrap'].rolling(window=window_size, center=True).max() - 
                               df['Yaw_unwrap'].rolling(window=window_size, center=True).min()).fillna(0)

            # Roll, Centripetal_Acc 모두 버리고 기하학적 각도 피처만 사용
            feature_cols = ['Yaw_rate', 'Yaw_range']
            df[feature_cols] = df[feature_cols].fillna(0)
            
            # 3. HMM Clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[feature_cols])
            
            from hmmlearn import hmm
            print("Using GaussianHMM for segmentation...")
            model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(X_scaled)
            df['State_Raw'] = model.predict(X_scaled)
            
            state_yaw_rates = []
            for s in range(2):
                state_mean = np.mean(X_scaled[df['State_Raw'] == s], axis=0)
                state_yaw_rates.append(state_mean[feature_cols.index('Yaw_rate')])
            turn_label = np.argmax(state_yaw_rates)

            df['State'] = df['State_Raw'].apply(lambda x: 1 if x == turn_label else 0)
            df['State_Smooth'] = df['State']
            
            df['Segment_Change'] = df['State_Smooth'].diff().fillna(0).abs()
            df['Final_Segment_ID'] = df['Segment_Change'].cumsum().astype(int)

            # 4. PCA 기반 직선성 정밀 검증
            from sklearn.decomposition import PCA as _PCA

            def is_linear_path(lats, lons, linearity_threshold=0.98):
                if len(lats) < 6:
                    return False, 0.0
                lat_rad_local = np.deg2rad(np.mean(lats))
                xs = np.deg2rad(lons) * R * np.cos(lat_rad_local)
                ys = np.deg2rad(lats) * R
                coords = np.column_stack([xs, ys])
                
                span = np.ptp(coords, axis=0)
                if np.max(span) < 1.0: 
                    return True, 1.0

                pca = _PCA(n_components=2)
                pca.fit(coords)
                ratio = pca.explained_variance_ratio_[0]
                return ratio >= linearity_threshold, ratio

            df['State_Final'] = df['State_Smooth'].copy()
            df['PCA_Ratio'] = 0.0

            rotate_seg_ids = df[df['State_Smooth'] == 1]['Final_Segment_ID'].unique()
            for rid in rotate_seg_ids:
                seg_mask = (df['Final_Segment_ID'] == rid)
                seg = df[seg_mask]
                lats = seg['Latitude'].values
                lons = seg['Longitude'].values

                gps_is_linear, pca_ratio = is_linear_path(lats, lons)
                df.loc[seg_mask, 'PCA_Ratio'] = pca_ratio

                if len(seg) > 0:
                    yaw_change = seg['Yaw_unwrap'].max() - seg['Yaw_unwrap'].min()
                else:
                    yaw_change = 0.0

                YAW_CHANGE_THRESHOLD = 45.0
                
                if gps_is_linear and yaw_change < YAW_CHANGE_THRESHOLD:
                    df.loc[seg_mask, 'State_Final'] = 3  # Error
                else:
                    if gps_is_linear:
                        df.loc[seg_mask, 'State_Final'] = 2  # Rotate_Line
                    else:
                        df.loc[seg_mask, 'State_Final'] = 1  # Normal Rotate

            df.loc[df['State_Smooth'] == 0, 'State_Final'] = 0
            df['Segment_Change_Final'] = (df['State_Final'] != df['State_Final'].shift(1)).astype(int)
            df['Final_Segment_ID'] = df['Segment_Change_Final'].cumsum().astype(int)

            # 5. 체인 병합 알고리즘
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
            ordered_segs = list(zip(seg_order_df['Final_Segment_ID'].tolist(), seg_order_df['state_final'].tolist()))

            runs = []
            current_run = []
            for seg_id, state in ordered_segs:
                if state in (0, 2, 3):
                    current_run.append((seg_id, state))
                else:
                    if current_run:
                        runs.append(current_run)
                        current_run = []
                    runs.append([(seg_id, state)])
            if current_run:
                runs.append(current_run)

            merge_ops = {}
            for run in runs:
                if len(run) == 1 and run[0][1] == 1:
                    continue
                line_positions = [i for i, (_, st) in enumerate(run) if st == 0]
                if not line_positions:
                    continue
                target_id = run[line_positions[0]][0]
                for old_id, _ in run:
                    if old_id != target_id:
                        merge_ops[old_id] = target_id

            n_merged = 0
            all_target_ids = set(merge_ops.values())
            for old_id, new_id in merge_ops.items():
                df.loc[df['Final_Segment_ID'] == old_id, 'Final_Segment_ID'] = new_id
                n_merged += 1

            for tid in all_target_ids:
                df.loc[(df['Final_Segment_ID'] == tid) & (df['State_Final'].isin([2, 3])), 'State_Final'] = 0

            if n_merged > 0:
                print(f"[Merge] {n_merged} segments merged into Line.")
                
            alive_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
            id_mapping = {old_id: new_idx for new_idx, old_id in enumerate(alive_seg_ids, start=1)}
            df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'] = df.loc[df['Final_Segment_ID'] != -1, 'Final_Segment_ID'].map(id_mapping).astype(int)
            
            # 6. 지도 시각화 및 다운로드 링크 생성
            output_file = os.path.join(temp_dir, "flight_path_segmented.html")
            output_split_dir = os.path.join(temp_dir, "splited")
            if not os.path.exists(output_split_dir):
                os.makedirs(output_split_dir)

            csv_links = []
            
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                center_lat = df['Latitude'].mean()
                center_lon = df['Longitude'].mean()
                import folium
                m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
                
                all_coords = df[['Latitude', 'Longitude']].values.tolist()
                folium.PolyLine(
                    locations=all_coords, color="#000000", weight=5, opacity=0.5, tooltip="Original Path"
                ).add_to(m)
                
                # 36색 팔레트 적용
                colors = [
                    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#0080FF', 
                    '#FF0080', '#80FF00', '#00FF80', '#FF4040', '#4040FF', '#FFD700', '#ADFF2F', '#FF69B4', 
                    '#1E90FF', '#DC143C', '#8B4513', '#006400', '#4B0082', '#008080', '#D2691E', '#7FFFD4',
                    '#FF1493', '#32CD32', '#00008B', '#B8860B', '#800000', '#9ACD32', '#20B2AA', '#E9967A',
                    '#9400D3', '#FF6600', '#DA70D6', '#2E8B57'
                ]
                
                sorted_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
                
                for seg_id in sorted_seg_ids:
                    group = df[df['Final_Segment_ID'] == seg_id]
                    coords = group[['Latitude', 'Longitude']].values.tolist()

                    seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]

                    if seg_state_final == 0:
                        type_str, label_prefix = "Straight", "Line"
                        color = colors[(seg_id - 1) % len(colors)]
                        weight, opacity, dash_array = 3, 0.8, None
                        link_style = "display:block; margin:5px 0; color:#333;"
                    elif seg_state_final == 1:
                        type_str, label_prefix = "Rotate", "Rotate"
                        color = colors[(seg_id - 1) % len(colors)]
                        weight, opacity, dash_array = 5, 0.9, '5, 5'
                        link_style = "display:block; margin:5px 0; color:#0055aa; font-weight:bold;"
                    elif seg_state_final == 2:
                        type_str, label_prefix = "Rotate_Line", "Rotate_Line"
                        color, weight, opacity, dash_array = '#9400D3', 5, 0.9, '10, 5'
                        link_style = "display:block; margin:5px 0; color:#9400D3; font-weight:bold;"
                    else:
                        type_str, label_prefix = "Rotate_Error", "Rotate_Error"
                        color, weight, opacity, dash_array = '#FF6600', 5, 0.9, '10, 5'
                        link_style = "display:block; margin:5px 0; color:#FF6600; font-weight:bold;"

                    label_text = f"{label_prefix}-{seg_id}"
                    if (seg_state_final != 0 )and 'PCA_Ratio' in group.columns:
                        pca_val = group['PCA_Ratio'].iloc[0]
                        label_text += f" (PCA: {pca_val:.3f})"

                    # 맵 시각화 추가
                    indices = group.index.values
                    if len(indices) > 0:
                        split_locs = np.where(np.diff(indices) > 1)[0] + 1
                        sub_groups_indices = np.split(indices, split_locs)
                        for sub_indices in sub_groups_indices:
                            if len(sub_indices) < 2: continue 
                            sub_coords = df.loc[sub_indices, ['Latitude', 'Longitude']].values.tolist()
                            folium.PolyLine(
                                locations=sub_coords, color=color, weight=weight, 
                                opacity=opacity, dash_array=dash_array, tooltip=f"ID {seg_id}: {type_str}"
                            ).add_to(m)
                            
                            folium.CircleMarker(
                                location=sub_coords[0], radius=3 if seg_state_final == 3 else 2,
                                color=color, fill=True, popup=f"ID {seg_id} [{type_str}] part start"
                            ).add_to(m)

                    # 마커 텍스트
                    center_lat_g, center_lon_g = group['Latitude'].mean(), group['Longitude'].mean()
                    if seg_state_final == 2:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 2px white;">🔄 {label_text}</div>'
                    elif seg_state_final == 3:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: #FF6600; text-shadow: 1px 1px 2px white;">⚠️ {label_text}</div>'
                    else:
                        label_html = f'<div style="font-size: 10pt; font-weight: bold; color: {color}; text-shadow: 1px 1px 1px white;">{label_text}</div>'

                    folium.Marker(
                        location=[center_lat_g, center_lon_g],
                        icon=folium.DivIcon(icon_size=(180, 36), icon_anchor=(0, 0), html=label_html)
                    ).add_to(m)

                    # CSV 생성 및 다운로드 링크 (원본 데이터 보존)
                    file_name = f"{label_prefix}-{seg_id}.csv"
                    save_path = os.path.join(output_split_dir, file_name)
                    
                    # 원본 df_orig에서 해당 인덱스들만 추출 (헤더와 인덱스 없이 원본 포맷 유지)
                    group_orig = df_orig.loc[group.index]
                    csv_content = group_orig.to_csv(index=False, header=False)
                    
                    with open(save_path, "w", encoding='utf-8') as f:
                        f.write(csv_content)

                    import base64
                    b64_csv = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
                    prefix_icon = "⚠️ " if seg_state_final == 3 else ("🔄 " if seg_state_final == 2 else "")
                    download_link = f'<a href="data:text/csv;base64,{b64_csv}" download="{file_name}" style="{link_style}">{prefix_icon}Download {file_name}</a>'
                    csv_links.append(download_link)

                m.save(output_file)
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                if csv_links:
                    download_panel = f"""
                    <div style="position: fixed; top: 10px; right: 10px; width: 250px; max-height: 80vh; overflow-y: auto; 
                                background-color: white; padding: 10px; border: 2px solid #ccc; z-index: 9999; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
                        <h4>Segment CSV Downloads</h4>
                        {''.join(csv_links)}
                    </div>
                    """
                    if "</body>" in html_content:
                        html_content = html_content.replace("</body>", f"{download_panel}</body>")
                    else:
                        html_content += download_panel

                return HTMLResponse(content=html_content)
                
        except Exception as e:
            print(f"오류 발생: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})







async def get_drone_analysis_stream(base64_image: str, context_json: str, mime_type: str = "image/png"):
    """
    GPT Vision 모델을 사용하여 드론 경로의 SPLIT/MERGE 오류를 분석하는 코어 스트리밍 로직입니다.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        split_text = ""
        merge_text = ""
        
        # --- PHASE 1: SPLIT 분석 ---
        yield "[THINKING]### [SPLIT (분리) 분석 결과]\n"
        split_payload = {
            "model": "gpt-5.4-2026-03-05", 
            "messages": [
                {"role": "system", "content": "당신은 전문 항공 데이터 분석가입니다."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        {
                            "type": "text",
                            "text": (
                                "### 임무 (TASK 1): SPLIT (분리) 분석\n"
                                "1. [분석 대상]: 'Line(직선)', 'Rotate_Line(보라색)', 'Rotate_Error(주황색)' 클러스터를 모두 검토하세요.\n"
                                "2. 클러스터 내부에 90도 이상의 급격한 회전, 굴곡(n-자, ㄴ-자, ㄷ-자 형태), 또는 루프가 포함된 경우를 찾아내세요.\n"
                                "3. 실제로는 명확한 직선과 커브,회전이 섞여 있어 분리가 필요한 케이스를 엄격하게 찾아내세요.\n"
                                "꺾이는 지점이 같은 색으로 묶여 있다면 이는 명백한 분리 오류입니다.\n\n"
                                "### 부정적 제약 조건 (중요 지침)\n"
                                "- 이미 코너 부분이 별도의 'Rotate' 클러스터로 분리되어 색상이 바뀌어 있다면 이는 정상입니다.\n"
                                "- 알고리즘이 턴(Turn) 구간을 감지하지 못해 두 개의 직선 변을 하나의 색으로 이어버린 모든 케이스를 보고하세요.\n"
                                "### 보고 형식 (REPORT FORMAT)\n"
                                "모든 답변은 '한국어'로 작성하세요. 식별된 각 SPLIT 예외 사항에 대해 대상 클러스터 ID와 구체적인 이유를 일반 텍스트 형식으로 설명해 주세요."
                            )
                        },
                    ]
                }
            ],
            "stream": True
        }

        async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                               headers={"Authorization": f"Bearer {OPENAPI_KEY}"}, json=split_payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if "[DONE]" in line: break
                    try:
                        content = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            split_text += content
                            yield content
                    except: continue

        # --- PHASE 2: MERGE 분석 ---
        yield "\n\n---\n### [MERGE (병합) 분석 결과]\n"
        merge_payload = {
            "model": "gpt-5.4-2026-03-05",
            "messages": [
                {"role": "system", "content": "당신은 전문 항공 데이터 분석가입니다."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        {
                            "type": "text",
                            "text": (
                                f"다음 드론 궤적 이미지와 컬러 매핑 정보를 분석해 주세요:\n{context_json}\n\n"
                                "### 임무 (TASK 2): MERGE (병합) 분석\n"
                                "1. [MERGE (병합)]: 동일한 성격의 경로가 아무런 이유 없이 두 개 이상의 인접한 색상으로 나뉜 경우를 찾아내세요.\n"
                                "2. [직선 병합]: 완벽하게 동일한 베이스라인(직선) 상에 있으며, 단순히 컬러만 중간에 바뀐 인접한 두 직선만 병합 대상으로 판단하세요.\n"
                                "3. [회전 병합]: 하나의 연속된 부드러운 곡선이나 루프가 여러 개의 Rotate 색상으로 쪼개져 있는 경우만 병합 대상으로 판단하세요.\n"
                                "4. [병합 금지]: 경로가 꺾이는 '꼭짓점(Corner)'을 기준으로 나뉜 구간은 절대 병합하면 안 됩니다. 꺾임이 있다면 서로 다른 성격의 구간입니다.\n"
                                "5. [병합 금지]: 두 직선(Line) 사이에 Rotate(회전) 세그먼트가 존재한다면, 이는 방향을 바꾸기 위한 의도적인 분할이므로 병합 대상에서 제외하세요.\n\n"
                                "모든 답변은 '한국어'로 작성하세요. 식별된 각 MERGE 예외 사항에 대해 대상 ID와 구체적인 이유를 설명해 주세요."
                            )
                        },
                    ]
                }
            ],
            "stream": True
        }

        async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                               headers={"Authorization": f"Bearer {OPENAPI_KEY}"}, json=merge_payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if "[DONE]" in line: break
                    try:
                        content = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            merge_text += content
                            yield content
                    except: continue

        # --- PHASE 3: 최종 요약 ---
        yield "[/THINKING][RESULT]"
        summary_payload = {
            "model": "gpt-5.4-2026-03-05",
            "messages": [
                {"role": "system", "content": "당신은 드론 경로 분석 전문가입니다. 핵심 결론만 아주 간결하게 리스트 형태로 요약하세요."},
                {
                    "role": "user",
                    "content": f"상세 분석 결과:\n\n[SPLIT 분석]\n{split_text}\n\n[MERGE 분석]\n{merge_text}\n\n위 내용을 바탕으로 최종 결론을 '한국어'로 작성해 주세요."
                }
            ],
            "stream": True
        }

        async with client.stream("POST", "https://api.openai.com/v1/chat/completions",
                               headers={"Authorization": f"Bearer {OPENAPI_KEY}"}, json=summary_payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if "[DONE]" in line: break
                    try:
                        content = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                        if content: yield content
                    except: continue

@app.post("/drone_analyze_gpt")
async def drone_analyze_gpt(file: UploadFile = File(...), context_json: str = Form(...)):
    if not OPENAPI_KEY:
        raise HTTPException(status_code=500, detail="OPENAI API key not configured")
    try:
        image_content = await file.read()
        import base64
        base64_image = base64.b64encode(image_content).decode('utf-8')
        mime_type = "image/png" if file.filename.endswith(".png") else "image/jpeg"
        return StreamingResponse(get_drone_analysis_stream(base64_image, context_json, mime_type), media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



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
            df_orig = pd.read_csv(target_file, header=None)
            df = df_orig.copy()
            
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

            df['State'] = df['State_Raw'].apply(lambda x: 1 if x == turn_label else 0)
            
            df['State_Smooth'] = df['State'] # Use Raw State
            
            # 5. Segment Identification & Merging (구간 나누기 및 병합) - Simplified
            df['Segment_Change'] = df['State_Smooth'].diff().fillna(0).abs()
            # Simple cumulative sum to create unique IDs for every state change
            df['Final_Segment_ID'] = df['Segment_Change'].cumsum().astype(int)
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
                    '#9400D3', # Violet
                    '#FF6600', # OrangeRed
                    '#008080', # Teal
                    '#800000', # Maroon
                    '#808000', # Olive
                    '#000080', # Navy
                    '#32CD32', # LimeGreen
                    '#FF7F50', # Coral
                    '#BA55D3', # MediumOrchid
                    '#20B2AA', # LightSeaGreen
                ]
                
                sorted_seg_ids = sorted(df[df['Final_Segment_ID'] != -1]['Final_Segment_ID'].unique())
                
                # 색상 이름 매핑 (VLM 최적화용)
                hex_to_color_name = {
                    '#FF0000': 'Red', '#00FF00': 'Lime', '#0000FF': 'Blue',
                    '#FFFF00': 'Yellow', '#FF00FF': 'Magenta', '#00FFFF': 'Cyan',
                    '#FF8000': 'Orange', '#0080FF': 'Azure', '#FF0080': 'DeepPink',
                    '#80FF00': 'Chartreuse', '#00FF80': 'SpringGreen', '#FF4040': 'LightRed',
                    '#4040FF': 'LightBlue', '#FFD700': 'Gold', '#ADFF2F': 'GreenYellow',
                    '#FF69B4': 'HotPink', '#1E90FF': 'DodgerBlue', '#DC143C': 'Crimson',
                    '#9400D3': 'Violet', '#FF6600': 'OrangeRed',
                    '#008080': 'Teal', '#800000': 'Maroon', '#808000': 'Olive',
                    '#000080': 'Navy', '#32CD32': 'LimeGreen', '#FF7F50': 'Coral',
                    '#BA55D3': 'MediumOrchid', '#20B2AA': 'LightSeaGreen'
                }
                
                cluster_color_map = {} # VML 분석을 위한 요약 맵

                for seg_id in sorted_seg_ids:
                    group = df[df['Final_Segment_ID'] == seg_id]
                    
                    # State_Final 기준으로 세그먼트 타입 결정
                    # 0=Straight, 1=Rotate, 2=Rotate_Line, 3=Rotate_Error
                    seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]

                    if seg_state_final == 0:
                        type_str = "Straight"
                        label_prefix = "Line"
                        color = colors[(seg_id - 1) % len(colors)]
                        weight = 3
                        opacity = 0.8
                        dash_array = None
                    elif seg_state_final == 1:
                        type_str = "Rotate"
                        label_prefix = "Rotate"
                        color = colors[(seg_id - 1) % len(colors)]
                        weight = 5
                        opacity = 0.9
                        dash_array = '5, 5'
                    elif seg_state_final == 2:
                        type_str = "Rotate_Line"
                        label_prefix = "Rotate_Line"
                        color = '#9400D3'  # Violet
                        weight = 5
                        opacity = 0.9
                        dash_array = '10, 5'
                    else:  # state == 3: Rotate_Error
                        type_str = "Rotate_Error"
                        label_prefix = "Rotate_Error"
                        color = '#FF6600'  # OrangeRed
                        weight = 5
                        opacity = 0.9
                        dash_array = '10, 5'

                    # VLM용 컬러 맵 업데이트
                    full_label = f"{label_prefix}-{seg_id}"
                    c_name = hex_to_color_name.get(color, color)
                    if c_name not in cluster_color_map:
                        cluster_color_map[c_name] = full_label
                    else:
                        cluster_color_map[c_name] += f", {full_label}"

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

                    # Generate CSV content (원본 데이터 보존)
                    group_orig = df_orig.loc[group.index]
                    csv_content = group_orig.to_csv(index=False, header=False)

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

                # --- 8. [새 기능] Matplotlib을 이용한 단순화된 시각화 이미지 생성 ---
                img_base64 = ""
                try:
                    import matplotlib.pyplot as plt
                    import io
                    plt.figure(figsize=(12, 8))
                    plt.style.use('default')
                    
                    # 배경 경로
                    plt.plot(df['Longitude'], df['Latitude'], color='#cccccc', linewidth=1, alpha=0.5, label="Original Path")
                    
                    # 각 세그먼트별 플로팅
                    for seg_id in sorted_seg_ids:
                        group = df[df['Final_Segment_ID'] == seg_id]
                        seg_state_final = group['State_Final'].iloc[0] if 'State_Final' in group.columns else group['State_Smooth'].iloc[0]

                        # 색상 및 스타일 설정 (Folium과 동일한 로직)
                        if seg_state_final == 0:
                            c = colors[(seg_id - 1) % len(colors)]
                            lw, ls = 2, '-'
                        elif seg_state_final == 1:
                            c = colors[(seg_id - 1) % len(colors)]
                            lw, ls = 3, '--'
                        elif seg_state_final == 2:
                            c, lw, ls = '#9400D3', 3, ':'
                        else:
                            c, lw, ls = '#FF6600', 3, '-.'

                        # 불연속 구간 처리
                        indices = group.index.values
                        if len(indices) > 0:
                            idx_diff = np.diff(indices)
                            split_locs = np.where(idx_diff > 1)[0] + 1
                            sub_groups_indices = np.split(indices, split_locs)

                            for sub_indices in sub_groups_indices:
                                if len(sub_indices) < 2: continue
                                sub_group = df.loc[sub_indices]
                                plt.plot(sub_group['Longitude'], sub_group['Latitude'], color=c, linewidth=lw, linestyle=ls)

                    plt.axis('off')
                    plt.tight_layout()
                    
                    # 이미지를 base64로 변환
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()
                except Exception as plt_e:
                    print(f"Matplotlib plotting error: {plt_e}")

                # --- 9. [새 기능] AI VLM 분석 자동 통합 ---
                ai_full_text = ""
                if OPENAPI_KEY:
                    print("Starting integrated AI analysis...")
                    async for chunk in get_drone_analysis_stream(img_base64, json.dumps(cluster_color_map, indent=1)):
                        # 스트리밍 태그 제거 후 텍스트만 취합
                        if chunk not in ("[THINKING]", "[/THINKING][RESULT]"):
                            ai_full_text += chunk
                
                # --- 사이드바 레이아웃 구성 ---
                sidebar_html = f"""
                <div id="sidebar" style="position: fixed; top: 0; right: 0; width: 400px; height: 100vh; 
                            background-color: white; border-left: 2px solid #ccc; z-index: 10000; 
                            overflow-y: auto; padding: 20px; box-sizing: border-box; font-family: sans-serif; box-shadow: -2px 0 10px rgba(0,0,0,0.1);">
                    <h3 style="margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 10px;">Analysis Report</h3>
                    
                    <div id="ai_report_section" style="margin-bottom: 20px; background: #fffbe6; padding: 15px; border: 1px solid #ffe58f; border-radius: 8px;">
                        <h4 style="margin: 0 0 10px 0; color: #856404;">🤖 AI Analysis Report</h4>
                        <div id="ai_report_content" style="font-size: 9pt; line-height: 1.6; white-space: pre-wrap;">
                            {ai_full_text if ai_full_text else "AI 분석 키가 설정되지 않았거나 분석에 실패했습니다."}
                        </div>
                    </div>

                    <div id="download_section">
                        <h4>Segment CSV Downloads</h4>
                        <div style="max-height: 200px; overflow-y: auto; border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                            {''.join(csv_links)}
                        </div>
                    </div>

                    <div style="margin-top: 20px; border-top: 2px solid #eee; padding-top: 15px;">
                        <h4 style="margin:0 0 10px 0;">AI Analyze Overview (Static View)</h4>
                        <img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto; border: 1px solid #eee; border-radius: 5px;" />
                        <div style="font-size: 8pt; color: #555; margin-top: 10px; background: #f9f9f9; padding: 8px; border-radius: 5px; border: 1px solid #eee;">
                            <strong>Color Mapping for AI:</strong><br/>
                            <pre id="vlm_color_mapping" style="white-space: pre-wrap; margin:0;">{json.dumps(cluster_color_map, indent=1)}</pre>
                        </div>
                    </div>
                </div>
                <style>
                    .folium-map {{ width: calc(100% - 400px) !important; height: 100vh !important; position: absolute !important; left: 0 !important; top: 0 !important; }}
                    #ai_report_content h3 {{ font-size: 11pt; margin: 15px 0 5px 0; border-bottom: 1px solid #ddd; }}
                    @media (max-width: 1000px) {{
                        #sidebar {{ width: 100%; height: auto; position: relative; border-left: none; border-top: 2px solid #ccc; }}
                        .folium-map {{ width: 100% !important; height: 50vh !important; position: relative !important; }}
                    }}
                </style>
                """
                
                if "</body>" in html_content:
                    html_content = html_content.replace("</body>", f"{sidebar_html}</body>")
                else:
                    html_content += sidebar_html

                return HTMLResponse(content=html_content)
                
        except Exception as e:
            print(f"오류 발생: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
