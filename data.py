import pandas as pd
import numpy as np
import time
from datetime import datetime
import FinanceDataReader as fdr # pykrx 대신 fdr 사용

import config

class RollingWindowScaler:
    def __init__(self):
        pass

    def rolling_window_scaling(self, df,window_len,columns = None):
        epsilon = 1e-8
        df = df.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if columns == None:
            columns = df.columns
        for col in columns:
            # 롤링 평균과 표준편차 계산
            roll_mean = df[col].rolling(window=window_len,min_periods=int(window_len/2) ).mean()
            roll_std = df[col].rolling(window=window_len,min_periods=int(window_len/2)).std()

            # Z-Score 변환 (표준편차가 0일 경우 대비 epsilon 추가)
            df[col] = (df[col] - roll_mean) / (roll_std + epsilon)

        return df
    def rolling_minmax(self, df, window_len, columns=None):
        # Min-Max 방식 (0~1 Scaling)
        epsilon = 1e-6
        df = df.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if columns is None:
            columns = df.columns

        for col in columns:
            # 롤링 최솟값과 최댓값 계산
            roll_min = df[col].rolling(window=window_len, min_periods=int(window_len/2)).min()
            roll_max = df[col].rolling(window=window_len, min_periods=int(window_len/2)).max()

            # Min-Max Scaling 공식: (X - Min) / (Max - Min)
            df[col] = (df[col] - roll_min) / (roll_max - roll_min + epsilon)

        return df

    def static_clip_minmax(self, df, columns=None, mini=10, maxi=60): #역사적으로 vix는 10~60사이를 오감
        df = df.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if columns is None:
            columns = df.columns

        for col in columns:
            # 1. 로그 변환 (하위 구간 해상도 확보)
            val_log = np.log1p(df[col])
            min_log = np.log1p(mini)
            max_log = np.log1p(maxi)

            # 2. 고정 Min-Max 스케일링
            scaled = (val_log - min_log) / (max_log - min_log)

            # 3. Clip (0~1 범위를 넘어서는 극단값 제한)
            df[col] = scaled.clip(0, 1)

        return df
    def ewm_window_scaling(self, df, window_len, ticker, columns=None, clip_limit=2):
        epsilon = 1e-8
        df = df.copy()

        if isinstance(df, pd.Series):
            df = df.to_frame()

        if columns is None:
            columns = df.columns

        # ---------------------------------------------------------
        # 1. OHLC 및 가격 기반 컬럼 식별
        # ---------------------------------------------------------
        # 종가 컬럼을 스케일링의 '기준(Anchor)'으로 삼습니다.
        close_col = f'{ticker}_log_close'

        # 종가의 스케일을 같이 적용받아야 하는 가격 관련 컬럼들
        # (만약 VWAP이나 SMA도 스케일링을 한다면 이 리스트에 포함해야 이격이 유지됩니다)
        price_cols = [f'{ticker}_log_open', f'{ticker}_log_high', f'{ticker}_log_low', f'{ticker}_log_close']

        # 현재 df에 존재하는 가격 컬럼만 필터링
        target_price_cols = [col for col in price_cols if col in columns and col in df.columns]

        # 그 외 개별적으로 스케일링할 컬럼들 (Volume, RSI, Frac_diff 등)
        other_cols = [col for col in columns if col not in target_price_cols + [f'{ticker}_vol_frac_diff',f'{ticker}_frac_diff'] and col in df.columns]

        # ---------------------------------------------------------
        # 2. 가격 컬럼 그룹 스케일링 (종가 기준 통일)
        # ---------------------------------------------------------
        if close_col in df.columns and len(target_price_cols) > 0:
            # 기준이 되는 종가의 EWM 평균과 표준편차를 구함
            close_ewm_mean = df[close_col].ewm(span=window_len, min_periods=int(window_len/2)).mean()
            close_ewm_std = df[close_col].ewm(span=window_len, min_periods=int(window_len/2)).std()

            for col in target_price_cols:
                # 모든 OHLC 가격을 종가의 평균/표준편차로 스케일링 -> 캔들 꼬리/몸통 비율 유지!
                df[col] = (df[col] - close_ewm_mean) / (close_ewm_std + epsilon)

                if clip_limit is not None:
                    df[col] = df[col].clip(lower=-clip_limit, upper=clip_limit)

        for col in [f'{ticker}_vol_frac_diff',f'{ticker}_frac_diff']:
            ewm_mean = df[col].ewm(span=window_len*10, min_periods=int(window_len*10/2)).mean()
            ewm_std = df[col].ewm(span=window_len*10, min_periods=int(window_len*10/2)).std()

            df[col] = (df[col] - ewm_mean) / (ewm_std + epsilon)

            if clip_limit is not None:
                df[col] = df[col].clip(lower=-clip_limit, upper=clip_limit)


        # ---------------------------------------------------------
        # 3. 나머지 개별 컬럼 스케일링 (기존 로직 유지)
        # ---------------------------------------------------------
        for col in other_cols:
            ewm_mean = df[col].ewm(span=window_len, min_periods=int(window_len/2)).mean()
            ewm_std = df[col].ewm(span=window_len, min_periods=int(window_len/2)).std()

            df[col] = (df[col] - ewm_mean) / (ewm_std + epsilon)

            if clip_limit is not None:
                df[col] = df[col].clip(lower=-clip_limit, upper=clip_limit)

        return df

    def rolling_robust(self, df, window_len, columns=None, clip_limit=3.0):
        epsilon = 1e-8
        df = df.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if columns is None:
            columns = df.columns

        for col in columns:
            roll_median = df[col].rolling(window=window_len, min_periods=int(window_len/2)).median()
            roll_q1 = df[col].rolling(window=window_len, min_periods=int(window_len/2)).quantile(0.25)
            roll_q3 = df[col].rolling(window=window_len, min_periods=int(window_len/2)).quantile(0.75)

            iqr = roll_q3 - roll_q1

            df[col] = (df[col] - roll_median) / (iqr + epsilon)

            if clip_limit is not None:
                df[col] = df[col].clip(-clip_limit, clip_limit)

        return df

    def __call__(self,df,window_len, columns = None):
        return self.rolling_robust(df,window_len,columns)

class CyclicalFeatureEncoder:
    def __init__(self, include_dow=True, include_month=True):
        """
        :param include_dow: 요일(Day of Week) 인코딩 포함 여부
        :param include_month: 월(Month) 인코딩 포함 여부
        """
        self.include_dow = include_dow
        self.include_month = include_month

    def __call__(self, df):
        df_encoded = df.copy()
        idx = df_encoded.index

        # 1. 요일 인코딩 (주기: 7일)
        if self.include_dow:
            dow = idx.dayofweek
            df_encoded['DOW_SIN'] = np.sin(2 * np.pi * dow / 7)
            df_encoded['DOW_COS'] = np.cos(2 * np.pi * dow / 7)

        # 2. 월 인코딩 (주기: 12개월)
        if self.include_month:
            # 1~12월을 0~11로 변환하여 계산
            month = idx.month - 1
            df_encoded['MON_SIN'] = np.sin(2 * np.pi * month / 12)
            df_encoded['MON_COS'] = np.cos(2 * np.pi * month / 12)

        return df_encoded

class StockFeatureProcessor:
    def __init__(self, window_ma=20, window_rsi=14, window_vol=5):
        self.window_ma = window_ma
        self.window_rsi = window_rsi
        self.window_vol = window_vol

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 1 - (1 / (1 + rs)) # 0~1 사이로 정규화

    def __call__(self, df, macro):
        df = df.copy()
        res = pd.DataFrame(index=df.index)

        # 1. 로그 수익률 (Log Return)
        res['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # 2. 시가 상대값 (Open relative to Prev Close) - 갭 상승/하락 확인
        res['rel_open'] = np.log(df['Open'] / df['Close'].shift(1))

        # 3. 고가 상대값 (High relative to Close) - 윗꼬리
        res['rel_high'] = np.log(df['High'] / df['Close'])

        # 4. 저가 상대값 (Low relative to Close) - 밑꼬리
        res['rel_low'] = np.log(df['Low'] / df['Close'])

        # 5. 로그 거래량 변화율
        # 거래량이 0인 경우를 대비해 아주 작은 값(eps)을 더함
        res['log_vol_chg'] = np.log((df['Volume'] + 1e-6) / (df['Volume'].shift(1) + 1e-6))

        # 6. 볼린저 밴드 %B (가격 위치 정보)
        ma = df['Close'].rolling(self.window_ma).mean()
        std = df['Close'].rolling(self.window_ma).std()
        res['bb_pct'] = (df['Close'] - (ma - 2 * std)) / (4 * std + 1e-6)

        # 7. RSI (상대강도지수) - 0~1 스케일
        res['rsi'] = self._calculate_rsi(df['Close'], self.window_rsi)

        # 8. ATR / Price (가격 대비 변동성 비율)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.window_rsi).mean()
        res['atr_ratio'] = atr / df['Close']

        # 9. 이동평균 이격도 (Price Disparity)
        res['ma_disparity'] = df['Close'] / (ma + 1e-6)

        # 10. 거래량 이격도 (Volume Disparity)
        vma = df['Volume'].rolling(self.window_vol).mean()
        res['vol_disparity'] = (df['Volume'] + 1e-6) / (vma + 1e-6)

        # 결측치 제거 (이동평균 등으로 인해 앞부분에 발생)
        res.dropna(inplace=True)

        #res['Close'] = df['Close']
        #res[['log_ret', 'Close']]
        if macro:
            res = res[['log_ret', 'log_vol_chg']]
        return res

class KRXDataCollector:
    def __init__(self, years=20, delay=0.3):
        self.years = years
        self.delay = delay
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = self._calculate_start_date()

    def _calculate_start_date(self):
        now = datetime.now()
        return f"{now.year - self.years}-{now.strftime('%m-%d')}"

    def _get_cleaned_df(self, ticker):

        df = fdr.DataReader(ticker, self.start_date, self.end_date)

        if df.empty:
            return df

        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']]

        df = df[df['Volume'] > 0]

        df = df.dropna()

        return df

    def collect_macro(self, macro_ticker_list):
        print(f"\n--- 매크로 데이터 수집 시작 (총 {len(macro_ticker_list)}개) ---")
        macro_df_list = []

        for ticker in macro_ticker_list:
            try:
                df = self._get_cleaned_df(ticker)

                if df is not None and not df.empty:
                    df = transform(df, macro=True)

                    if not df.empty:
                        df.columns = [f"M_{ticker}_{col}" for col in df.columns]
                        macro_df_list.append(df)
                        print(f"매크로 [M_{ticker}] 수집 완료...")

                time.sleep(self.delay)

            except Exception as e:
                print(f"\n [매크로 {ticker} 건너뜀] 사유: {str(e)}")
                continue

        if not macro_df_list:
            return pd.DataFrame()

        combined_macro = pd.concat(macro_df_list, axis=1, join='outer')
        combined_macro = combined_macro.ffill().fillna(0)
        print(f"\n매크로 통합 완료: {combined_macro.shape}")
        return combined_macro

    def __call__(self, micro_ticker_list, macro_ticker_list):
        master_macro_df = self.collect_macro(macro_ticker_list)

        stock_dict = {}
        total = len(micro_ticker_list)
        print(f"\n--- 종목 데이터 수집 및 매크로 병합 시작 ---")

        for i, ticker in enumerate(micro_ticker_list):
            try:
                print(f"[{i+1}/{total}] {ticker} 처리 중...")

                df = self._get_cleaned_df(ticker)

                if df.empty:
                    continue

                df = transform(df, macro=False)

                combined_df = df.join(master_macro_df, how='outer')
                combined_df = combined_df.ffill().dropna()

                combined_df = roller(combined_df, window_len=INPUT_WINDOW)
                combined_df = sin_cos_encoder(combined_df)

                combined_df = combined_df.dropna()

                if not combined_df.empty:
                    stock_dict[ticker] = combined_df

                time.sleep(self.delay)
            except Exception as e:
                print(f"\n {ticker} 오류 발생: {e}")
                continue

        print(f"\n 최종 수집 완료: 총 {len(stock_dict)}개 종목 데이터 확보")
        return stock_dict

roller = RollingWindowScaler() # df, window_len
sin_cos_encoder = CyclicalFeatureEncoder(include_dow=config.INCLUDE_DOW , include_month=config.INCLUDE_MONTH) # df
transform = StockFeatureProcessor() # df, macro = True
