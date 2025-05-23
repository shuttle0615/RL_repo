import gymnasium as gym
import numpy as np
import pandas as pd

class BitcoinTradingEnv(gym.Env):
    """
    비트코인 트레이딩을 위한 Gymnasium 환경 (단순 수익률 모델).
    이 환경은 실제 포트폴리오 잔고를 추적하지 않고,
    이상적인 전액 투자 상황에서의 로그 수익률을 보상으로 계산합니다.

    Observation:
        - market_data: (window_size, 15) 크기의 배열 (OHLCV + 기술 지표).
        - position: 현재 포지션 (0: Short, 1: Neutral, 2: Long).

    Action:
        - 0: Short 포지션 진입 또는 유지.
        - 1: Neutral 포지션 (청산) 진입 또는 유지.
        - 2: Long 포지션 진입 또는 유지.
    """
    metadata = {'render_modes': ['human', None]}
    VALID_MODES = ['train', 'val', 'test']

    def __init__(self, csv_path, window_size=168, episode_length=1000,
                 transaction_fee_percent=0.001,
                 timestamp_col_name='Date', date_ranges_by_mode=None):
        super().__init__()

        self.window_size = window_size
        self.episode_length = episode_length
        self.transaction_fee_percent = float(transaction_fee_percent)
        
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"지정된 CSV 파일을 찾을 수 없습니다: {csv_path}")

        if self.timestamp_col_name not in self.df.columns:
            raise ValueError(f"CSV 파일에 지정된 시간 컬럼 '{self.timestamp_col_name}'이 없습니다.")
        try:
            self.df[self.timestamp_col_name] = pd.to_datetime(self.df[self.timestamp_col_name])
        except Exception as e:
            raise ValueError(f"날짜 컬럼 '{self.timestamp_col_name}'을 datetime으로 변환 중 오류 발생: {e}")

        self.ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in self.ohlcv_columns:
            assert col in self.df.columns, f"CSV 파일에 '{col}' 컬럼이 없습니다."

        # --- Placeholder: 기술적 지표 생성 ---
        # TODO:
        self.ti_column_names = [f'TI{i+1}' for i in range(10)]
        for ti_col in self.ti_column_names:
            if ti_col not in self.df.columns:
                 self.df[ti_col] = np.random.randn(len(self.df)) 
        # --- 기술적 지표 생성 완료 ---

        self.market_feature_columns = self.ohlcv_columns + self.ti_column_names
        
        self.action_space = gym.spaces.Discrete(3) # 0: Short, 1: Neutral, 2: Long
        self.observation_space = gym.spaces.Dict({
            "market_data": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, 15), dtype=np.float64
            ),
            "position": gym.spaces.Discrete(3)
        })

        self.current_position = 1
        self.current_df_idx = 0
        self.steps_this_episode = 0
        self.render_mode = 'None'
        self.log_fee = np.log(1 - self.transaction_fee_percent)
        self.terminated = False

        self.mode_indices = {}
        if not isinstance(date_ranges_by_mode, dict):
            raise TypeError("date_ranges_by_mode는 딕셔너리 타입이어야 합니다.")
            
        for mode_key in self.VALID_MODES:
            if mode_key not in date_ranges_by_mode:
                raise ValueError(f"date_ranges_by_mode가 제공되었지만 '{mode_key}' 모드에 대한 날짜 범위가 없습니다.")
            
            date_range_for_mode = date_ranges_by_mode[mode_key]
            if not (isinstance(date_range_for_mode, tuple) and len(date_range_for_mode) == 2):
                raise ValueError(f"'{mode_key}' 모드의 날짜 범위는 (시작일_문자열, 종료일_문자열) 형태의 튜플이어야 합니다.")

            start_date_str, end_date_str = date_range_for_mode
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            
            mode_df_slice = self.df[(self.df[self.date_column_name] >= start_date) & \
                                    (self.df[self.date_column_name] <= end_date)]
            if mode_df_slice.empty:
                raise ValueError(f"'{mode_key}' 모드에 대한 데이터가 지정된 날짜 범위 내에 없습니다: {start_date_str} ~ {end_date_str}")
            self.mode_indices[mode_key] = (mode_df_slice.index[0], mode_df_slice.index[-1])
        
        self.active_mode_start_idx = 0
        self.active_mode_end_idx = len(self.df) - 1


    def _get_current_price(self):
        return self.df.loc[self.current_df_idx, 'Close']

    def _get_obs(self):
        start_idx = self.current_df_idx - self.window_size + 1
        end_idx = self.current_df_idx + 1
        df_window = self.df.iloc[start_idx:end_idx].copy()

        first_close_in_window = df_window['Close'].iloc[0]
        first_volume_in_window = df_window['Volume'].iloc[0]

        for col in self.ohlcv_columns:
            if col == 'Volume':
                df_window[col] = np.log((df_window[col] + 1e-7) / (first_volume_in_window + 1e-7))
            else:
                df_window[col] = np.log(df_window[col] / (first_close_in_window + 1e-9))
        
        market_data_obs = df_window[self.market_feature_columns].values.astype(np.float64)
        
        return {
            "market_data": market_data_obs,
            "position": self.current_position
        }

    def reset(self, seed=None, options=None, mode='train'):
        super().reset(seed=seed)

        if mode not in self.VALID_MODES:
            raise ValueError(f"잘못된 모드 '{mode}'입니다. 허용되는 모드: {self.VALID_MODES}")

        self.current_position = 1 
        self.steps_this_episode = 0

        self.active_mode_start_idx, self.active_mode_end_idx = self.mode_indices[mode]
        
        earliest_possible_idx = max(self.window_size-1, self.active_mode_start_idx)

        latest_possible_idx = self.active_mode_end_idx - self.episode_length

        if earliest_possible_idx > latest_possible_idx:
            raise ValueError(
                f"'{mode}' 모드의 데이터가 너무 짧아 (유효 시작점 범위 없음) 에피소드를 시작할 수 없습니다. "
                f"계산된 시작/종료 가능 인덱스: {earliest_possible_idx} > {latest_possible_idx}"
            )

        # 유효한 범위 내에서 첫 관찰 윈도우의 마지막 날짜를 랜덤하게 선택
        self.current_df_idx = self.np_random.integers(
            earliest_possible_idx, latest_possible_idx + 1
        )
        
        obs = self._get_obs()
        info = { "action_taken_description": f"Reset to mode '{mode}'" }
        return obs, info

    def step(self, action):
        if self.terminated:
            raise RuntimeError("환경이 종료되었습니다. reset()을 호출하여 새 에피소드를 시작하세요.")

        previous_price = self._get_current_price()
        previous_position = self.current_position
        
        self.current_position = action

        self.current_df_idx += 1
        self.steps_this_episode += 1
        current_price = self._get_current_price()

        # 보상 계산 로직
        reward = 0.0

        # 1. 거래 수수료
        if self.current_position != previous_position:
            reward += self.log_fee
            is_reversal = (previous_position == 2 and self.current_position == 0) or \
                          (previous_position == 0 and self.current_position == 2)
            if is_reversal:
                reward += self.log_fee

        # 2. 가격 변동에 따른 수익률 (이전 스텝에서 포지션을 유지하고 있었을 때, terminated가 아닌 경우)
        price_ratio = current_price / previous_price

        if previous_position == 2: # Long 포지션이었다면
            # 가격이 0이 되는 경우 log(0)을 피하기 위해 max 사용
            reward += np.log(price_ratio)
        elif previous_position == 0: # Short 포지션이었다면
            arg_for_short_log = 2.0 - price_ratio
            # Short 포지션 손실이 100% (가격 2배 상승) 이상일 경우 log 인자가 0 이하가 될 수 있음
            reward += np.log(max(1e-9, arg_for_short_log))
        
        self.terminated = self.steps_this_episode >= self.episode_length
        
        obs = self._get_obs()
        info = { "reward": reward, "current_price": current_price }

        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, self.terminated, info

    def render(self):
        if self.render_mode == 'human':
            position_map = {0: "Short", 1: "Neutral", 2: "Long"}
            print(f"스텝: {self.steps_this_episode}/{self.episode_length} | "
                  f"포지션: {position_map[self.current_position]} | "
                  f"현재 가격: {self._get_current_price():.2f}")

    def close(self):
        pass

    def set_render_mode(self, mode):
        self.render_mode = mode
