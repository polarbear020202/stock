import pandas as pd
import numpy as np

class DfDictIterator:
    def __init__(self):
        pass

    def __call__(self, df_dict, func):
            new_df_dict = {}

            for ticker, df in df_dict.items():
                processed_df = func(df)
                new_df_dict[ticker] = processed_df
            return new_df_dict


class labeler:
    def __init__(self, window=10, ret_col='log_ret', min_return=0.05, max_time=30):
        """
        window: 고점/저점을 판단할 좌우 윈도우 크기
        ret_col: 기준이 되는 '로그 수익률' 컬럼명
        min_return: 상승/하락으로 인정할 최소 구간 로그 수익률 (기본값 3%)
        max_time: 고점/저점까지 도달하는 데 허용되는 최대 캔들 수 (기본값 20봉)
        """
        self.window = window
        self.ret_col = ret_col
        self.min_return = min_return
        self.max_time = max_time

    def __call__(self, df):
        df_temp = df.copy()

        # 1. 누적 수익률(가상 가격 궤적) 계산 -> 처음에 작성하셨던 훌륭한 로직 적용
        df_temp['cum_ret'] = df_temp[self.ret_col].cumsum()

        # 2. 로컬 고점/저점 마킹 (누적 수익률 기준)
        roll = df_temp['cum_ret'].rolling(window=self.window * 2 + 1, center=True)
        is_peak = df_temp['cum_ret'] == roll.max()
        is_trough = df_temp['cum_ret'] == roll.min()

        # 필터링하기 전에 원본 데이터에 미리 type을 계산해 둡니다. (고점 1, 저점 -1, 나머지 0)
        df_temp['type'] = np.where(is_peak, 1, np.where(is_trough, -1, 0))

        # 그 다음 0이 아닌(고점이나 저점인) 행들만 쏙 뽑아냅니다. (길이 불일치 원천 차단)
        events = df_temp[df_temp['type'] != 0].copy()

        # 3. Strict Alternation (고점/저점 교대 필터링)
        filtered_events = []
        last_type = 0

        for idx, row in events.iterrows():
            curr_type = row['type']
            curr_val = row['cum_ret']  # 비교 기준도 누적 수익률

            if last_type == 0:
                last_type = curr_type
                filtered_events.append({'idx': idx, 'type': curr_type, 'val': curr_val})
                continue

            if curr_type == last_type:
                last_event = filtered_events[-1]
                if (curr_type == 1 and curr_val > last_event['val']) or \
                   (curr_type == -1 and curr_val < last_event['val']):
                    filtered_events[-1] = {'idx': idx, 'type': curr_type, 'val': curr_val}
            else:
                last_type = curr_type
                filtered_events.append({'idx': idx, 'type': curr_type, 'val': curr_val})

        # 이벤트가 없으면 빈 타겟 리턴
        if not filtered_events:
            df_temp['target'] = np.nan
            return df_temp

        events_df = pd.DataFrame(filtered_events).set_index('idx')

        # 4. 다음 고점/저점 정보를 과거 방향으로 채우기 (Backward Fill)
        df_temp['temp_next_type'] = np.nan
        df_temp['temp_next_val'] = np.nan

        df_temp.loc[events_df.index, 'temp_next_type'] = events_df['type']
        df_temp.loc[events_df.index, 'temp_next_val'] = events_df['val']

        df_temp['temp_next_type'] = df_temp['temp_next_type'].bfill()
        df_temp['temp_next_val'] = df_temp['temp_next_val'].bfill()

        df_temp['temp_curr_pos'] = np.arange(len(df_temp))
        df_temp['temp_next_pos'] = np.nan
        df_temp.loc[events_df.index, 'temp_next_pos'] = df_temp.loc[events_df.index, 'temp_curr_pos']
        df_temp['temp_next_pos'] = df_temp['temp_next_pos'].bfill()

        # 5. 내부 조건 계산용 3가지 피처 (메모리상에서만 계산)
        temp_dir = df_temp['temp_next_type']
        temp_time = df_temp['temp_next_pos'] - df_temp['temp_curr_pos']

        # 💡 핵심: 로그 수익률이므로 (도착지 누적수익률 - 현재 누적수익률) = 미래에 얻을 구간 수익률
        temp_ret = df_temp['temp_next_val'] - df_temp['cum_ret']

        # 6. 분류용 타겟 (target) 생성
        df_temp['target'] = np.nan

        valid_mask = temp_dir.notna()

        # 일단 유효한 구간은 전부 1(횡보/관망)로 기본 세팅
        df_temp.loc[valid_mask, 'target'] = 1

        # 상승장(2) 조건: 다음이 고점(1) & 기대 수익률 >= 최소 조건 & 걸리는 시간 <= 최대 조건
        cond_long = valid_mask & (temp_dir == 1) & (temp_ret >= self.min_return) & (temp_time <= self.max_time)

        # 하락장(0) 조건: 다음이 저점(-1) & 기대 하락폭 <= 최소 조건 (음수) & 걸리는 시간 <= 최대 조건
        cond_short = valid_mask & (temp_dir == -1) & (temp_ret <= -self.min_return) & (temp_time <= self.max_time)

        # 조건에 맞는 부분만 덮어씌우기
        df_temp.loc[cond_long, 'target'] = 2
        df_temp.loc[cond_short, 'target'] = 0

        # 7. 라벨링 안 된 끝부분 삭제 및 불필요한 임시 컬럼 모두 제거
        df_temp.dropna(subset=['target'], inplace=True)
        df_temp['target'] = df_temp['target'].astype(int)

        # 처음에 계산했던 cum_ret도 깔끔하게 날려버립니다.
        drop_cols = ['cum_ret', 'temp_next_type', 'temp_next_val', 'temp_curr_pos', 'temp_next_pos','type']
        df_temp.drop(columns=drop_cols, inplace=True)

        return df_temp
