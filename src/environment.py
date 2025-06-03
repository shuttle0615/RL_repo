import gymnasium as gym
import numpy as np
import pandas as pd
import talib as ta
from tqdm import tqdm

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
                 transaction_fee_percent=0.0005,
                 timestamp_col_name='timestamp', date_ranges_by_mode=None):
        super().__init__()

        self.window_size = window_size
        self.episode_length = episode_length
        self.transaction_fee_percent = float(transaction_fee_percent)
        self.timestamp_col_name = timestamp_col_name
        
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

        # # --- Placeholder: 기술적 지표 생성 ---
        # # TODO:
        # self.ti_column_names = [f'TI{i+1}' for i in range(10)]
        # for ti_col in self.ti_column_names:
        #     if ti_col not in self.df.columns:
        #          self.df[ti_col] = np.random.randn(len(self.df)) 
        # # --- 기술적 지표 생성 완료 ---

        self.market_feature_columns = self.ohlcv_columns + self.generate_technical_indicators()
        
        flip_ti_cols = [        # indicators that change sign if trend is inverted
        'RSI', 'MACD', 'BB_Position',       # BB_Position 0↔1, but z-score makes sign arbitrary
        'OBV', 'Stoch_D', 'CCI', 'MFI', 'VWAP'
        ]
        no_flip_cols = ['Volume', 'ATR', 'ADX']  # magnitudes only
        self._flip_idx = [self.market_feature_columns.index(c)
                  for c in self.ohlcv_columns + flip_ti_cols]

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
        self.flip_sign = False

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
            
            mode_df_slice = self.df[(self.df[self.timestamp_col_name] >= start_date) & \
                                    (self.df[self.timestamp_col_name] <= end_date)]
            if mode_df_slice.empty:
                raise ValueError(f"'{mode_key}' 모드에 대한 데이터가 지정된 날짜 범위 내에 없습니다: {start_date_str} ~ {end_date_str}")
            self.mode_indices[mode_key] = (mode_df_slice.index[0], mode_df_slice.index[-1])
        
        self.active_mode_start_idx = 0
        self.active_mode_end_idx = len(self.df) - 1


    def _get_current_price(self):
        return self.df.loc[self.current_df_idx, 'Close']

    def _get_obs(self):
        """Return one observation window with per-column z-score normalisation
        for all technical indicators, while keeping price normalisation
        (log return to first close) unchanged.
        """
        # ----------------  slice the rolling window  -----------------
        start_idx = self.current_df_idx - self.window_size + 1
        end_idx   = self.current_df_idx + 1
        df_window = self.df.iloc[start_idx:end_idx].copy()

        # ----------------  price / volume normalisation --------------
        first_close  = df_window['Close'].iloc[0]
        first_volume = df_window['Volume'].iloc[0]

        for col in self.ohlcv_columns:
            if col == 'Volume':
                df_window[col] = np.log((df_window[col] + 1e-7) /
                                        (first_volume            + 1e-7))
            else:  # Open, High, Low, Close
                df_window[col] = np.log(df_window[col] / (first_close + 1e-9))

        # ----------------  per-indicator z-score ---------------------
        ti_cols = [c for c in self.market_feature_columns if c not in self.ohlcv_columns]
        if ti_cols:  # defensive: in case you switch features later
            mu  = df_window[ti_cols].mean()
            std = df_window[ti_cols].std().replace(0.0, 1.0)  # avoid 0-div
            df_window[ti_cols] = (df_window[ti_cols] - mu) / (std + 1e-8)

        # ----------------  assemble numpy array ----------------------
        market_data_obs = df_window[self.market_feature_columns].values.astype(np.float64)

        # ----------------  optional episode-level sign flip ----------
        if self.flip_sign:
            market_data_obs[:, self._flip_idx] *= -1.0

        # ----------------  return dict observation -------------------
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
        self.terminated = False

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
        
        #print(f"possible_idx: {earliest_possible_idx} ~ {latest_possible_idx}")
        # if mode == 'train':
        #     self.flip_sign = self.np_random.random() < 0.5

        obs = self._get_obs()
        info = { "action_taken_description": f"Reset to mode '{mode}'" }
        return obs, info

    def step(self, action, mode='train'):
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
        
        # if mode == 'train':
        #     reward = reward - np.log(price_ratio)  
        #     if previous_position == 1:        # flat/ cash
        #         reward -= 2e-4    
        #     elif previous_position == 2: # Long 포지션이었다면
        #         reward -= 4e-4
        # else:
        #     reward = reward  

        self.terminated = self.steps_this_episode >= self.episode_length
        
        obs = self._get_obs()
        info = { "reward": reward, "current_price": current_price }

        if self.render_mode == 'human':
            self.render()

        if self.flip_sign:
            reward = -reward    
        
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

    def generate_technical_indicators(self):
        """Generate 10 technical indicators using TA-Lib"""
        
        # Get OHLCV data
        open = self.df['Open'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        close = self.df['Close'].values
        volume = self.df['Volume'].values
        
        # 1. RSI (Relative Strength Index) - Momentum
        self.df['RSI'] = ta.RSI(close, timeperiod=14)
        
        # 2. MACD (Moving Average Convergence Divergence) - Trend
        macd, macd_signal, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD'] = macd - macd_signal  # MACD histogram
        
        # 3. Bollinger Bands - Volatility
        upper, middle, lower = ta.BBANDS(close, timeperiod=20)
        self.df['BB_Position'] = (close - lower) / (upper - lower)  # Position within BB
        
        # 4. ATR (Average True Range) - Volatility
        self.df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        
        # 5. OBV (On Balance Volume) - Volume
        self.df['OBV'] = ta.OBV(close, volume)
        
        # 6. ADX (Average Directional Index) - Trend Strength
        self.df['ADX'] = ta.ADX(high, low, close, timeperiod=14)
        
        # 7. Stochastic Oscillator - Momentum
        _, self.df['Stoch_D'] = ta.STOCH(high, low, close, 
                                        fastk_period=14, slowk_period=3, slowd_period=3)
        
        # 8. CCI (Commodity Channel Index) - Momentum/Overbought/Oversold
        self.df['CCI'] = ta.CCI(high, low, close, timeperiod=20)
        
        # 9. MFI (Money Flow Index) - Volume/Momentum
        self.df['MFI'] = ta.MFI(high, low, close, volume, timeperiod=14)
        
        # 10. VWAP (Volume Weighted Average Price) - Volume/Price
        self.df['VWAP'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        
        # Normalize indicators to similar scales
        # def normalize(series):
        #     return (series - series.rolling(window=100, min_periods=1).mean()) / \
        #            (series.rolling(window=100, min_periods=1).std() + 1e-8)
        
        indicators = ['RSI', 'MACD', 'BB_Position', 'ATR', 'OBV', 
                     'ADX', 'Stoch_D', 'CCI', 'MFI', 'VWAP']
        
        # for ind in indicators:
        #     self.df[ind] = normalize(self.df[ind])
            
        # Handle NaN values
        self.df = self.df.fillna(method='bfill').fillna(0)

        return indicators

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test configuration
    date_ranges = {
        'train': ('2019-09-01', '2022-12-31'),
        'val': ('2023-01-01', '2023-06-30'),
        'test': ('2023-07-01', '2023-12-31')
    }
    
    # Initialize environment
    env = BitcoinTradingEnv(
        csv_path="data/BTCUSDT_1h_full_history.csv",
        window_size=168,
        episode_length=168,
        date_ranges_by_mode=date_ranges
    )
    
    # Random policy test
    rng = np.random.default_rng(42)
    rewards = []
    episode_rewards = []
    current_episode_reward = 0
    
    print("Testing random policy...")
    state, _ = env.reset()
    
    for step in tqdm(range(50000)):
        action = rng.integers(0, 3)  # uniform random
        _, reward, done, _ = env.step(int(action))
        rewards.append(reward)
        current_episode_reward += reward
        
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            state, _ = env.reset()
    
    # Print statistics
    print("\nRandom Policy Statistics:")
    print(f"Mean step reward: {np.mean(rewards):.6f}")
    print(f"Std step reward: {np.std(rewards):.6f}")
    print(f"Mean episode return: {np.mean(episode_rewards):.6f}")
    print(f"Std episode return: {np.std(episode_rewards):.6f}")
    print(f"Mean episode return (exp): {np.exp(np.mean(episode_rewards)):.6f}")
    
    # Plot reward distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(rewards, bins=50, alpha=0.75)
    plt.title('Step Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=50, alpha=0.75)
    plt.title('Episode Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('random_policy_rewards.png')
    plt.close()
    
    # Plot cumulative returns
    plt.figure(figsize=(10, 5))
    cumulative_returns = np.cumsum(episode_rewards)
    plt.plot(cumulative_returns)
    plt.title('Cumulative Returns over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig('random_policy_cumulative.png')
    plt.close()
