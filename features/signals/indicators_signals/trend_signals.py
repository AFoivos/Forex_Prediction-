class SMASignalGenerator:
    """
    GENERATES TRADING SIGNALS USING ONLY SMA AND EMA INDICATORS
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
        
    # ==================== CROSSOVER SIGNALS ====================
    
    def golden_death_cross(self):
        """Golden Cross (SMA50 > SMA200) and Death Cross (SMA50 < SMA200)"""
        if 'trend_sma_50' in self.data.columns and 'trend_sma_200' in self.data.columns:
            # Golden Cross: SMA50 crosses above SMA200
            self.signals['golden_cross'] = (
                (self.data['trend_sma_50'] > self.data['trend_sma_200']) & 
                (self.data['trend_sma_50'].shift(1) <= self.data['trend_sma_200'].shift(1))
            )
            # Death Cross: SMA50 crosses below SMA200  
            self.signals['death_cross'] = (
                (self.data['trend_sma_50'] < self.data['trend_sma_200']) & 
                (self.data['trend_sma_50'].shift(1) >= self.data['trend_sma_200'].shift(1))
            )
        return self.signals[['golden_cross', 'death_cross']]
    
    def ema_crossover(self, fast_period=20, slow_period=50):
        """EMA Crossover signals (Bullish: EMA20 > EMA50, Bearish: EMA20 < EMA50)"""
        fast_col = f'trend_ema_{fast_period}'
        slow_col = f'trend_ema_{slow_period}'
        
        if fast_col in self.data.columns and slow_col in self.data.columns:
            # Bullish Crossover: Fast EMA crosses above Slow EMA
            self.signals[f'ema_bullish_{fast_period}_{slow_period}'] = (
                (self.data[fast_col] > self.data[slow_col]) & 
                (self.data[fast_col].shift(1) <= self.data[slow_col].shift(1))
            )
            # Bearish Crossover: Fast EMA crosses below Slow EMA
            self.signals[f'ema_bearish_{fast_period}_{slow_period}'] = (
                (self.data[fast_col] < self.data[slow_col]) & 
                (self.data[fast_col].shift(1) >= self.data[slow_col].shift(1))
            )
        return self.signals[[f'ema_bullish_{fast_period}_{slow_period}', 
                           f'ema_bearish_{fast_period}_{slow_period}']]
    
    # ==================== TREND HIERARCHY SIGNALS ====================
    
    def trend_hierarchy(self, periods=[10, 20, 50]):
        """Checks if EMAs are in perfect bullish/bearish alignment"""
        ema_cols = [f'trend_ema_{p}' for p in periods]
        
        # Verify all required columns exist
        if all(col in self.data.columns for col in ema_cols):
            # Bullish hierarchy: EMA10 > EMA20 > EMA50
            bullish_condition = True
            for i in range(len(ema_cols)-1):
                bullish_condition &= (self.data[ema_cols[i]] > self.data[ema_cols[i+1]])
            
            # Bearish hierarchy: EMA10 < EMA20 < EMA50  
            bearish_condition = True
            for i in range(len(ema_cols)-1):
                bearish_condition &= (self.data[ema_cols[i]] < self.data[ema_cols[i+1]])
            
            self.signals['trend_hierarchy_bullish'] = bullish_condition
            self.signals['trend_hierarchy_bearish'] = bearish_condition
        
        return self.signals[['trend_hierarchy_bullish', 'trend_hierarchy_bearish']]
    
    # ==================== SUPPORT/RESISTANCE BOUNCE SIGNALS ====================
    
    def ma_bounce_signals(self, ma_period=20, ma_type='ema'):
        """Signals when price bounces off moving average support/resistance"""
        ma_col = f'trend_{ma_type}_{ma_period}'
        
        if ma_col in self.data.columns:
            # Price touches or comes very close to MA (within 0.1%)
            touch_threshold = self.data[ma_col] * 0.001
            price_touches_ma = abs(self.data['close'] - self.data[ma_col]) <= touch_threshold
            
            # Bullish bounce: Price was below, touches MA, then closes above
            self.signals[f'{ma_type}_{ma_period}_bullish_bounce'] = (
                (self.data['close'].shift(1) < self.data[ma_col].shift(1)) &
                price_touches_ma &
                (self.data['close'] > self.data[ma_col])
            )
            
            # Bearish bounce: Price was above, touches MA, then closes below  
            self.signals[f'{ma_type}_{ma_period}_bearish_bounce'] = (
                (self.data['close'].shift(1) > self.data[ma_col].shift(1)) &
                price_touches_ma &
                (self.data['close'] < self.data[ma_col])
            )
        
        return self.signals[[f'{ma_type}_{ma_period}_bullish_bounce', 
                           f'{ma_type}_{ma_period}_bearish_bounce']]
    
    # ==================== SLOPE AND MOMENTUM SIGNALS ====================
    
    def ma_slope_signals(self, period=20, ma_type='ema', lookback=3):
        """Signals based on moving average slope and acceleration"""
        ma_col = f'trend_{ma_type}_{period}'
        slope_col = f'{ma_col}_slope'
        
        if ma_col in self.data.columns:
            # Create slope if it doesn't exist
            if slope_col not in self.data.columns:
                self.data[slope_col] = self.data[ma_col].diff()
            
            # Positive slope (uptrend)
            self.signals[f'{ma_type}_{period}_slope_positive'] = self.data[slope_col] > 0
            
            # Slope acceleration (slope increasing)
            self.signals[f'{ma_type}_{period}_slope_accelerating'] = (
                self.data[slope_col] > self.data[slope_col].shift(1)
            )
            
            # Strong trend: Price above MA and MA slope positive
            self.signals[f'{ma_type}_{period}_strong_uptrend'] = (
                (self.data['close'] > self.data[ma_col]) & 
                (self.data[slope_col] > 0)
            )
        
        return self.signals[[f'{ma_type}_{period}_slope_positive',
                           f'{ma_type}_{period}_slope_accelerating',
                           f'{ma_type}_{period}_strong_uptrend']]
    
    # ==================== PRICE EXTENSION SIGNALS ====================
    
    def price_extension_signals(self, ma_period=20, ma_type='ema', deviation=0.02):
        """Signals when price is overextended from moving average"""
        ma_col = f'trend_{ma_type}_{ma_period}'
        
        if ma_col in self.data.columns:
            # Calculate percentage deviation from MA
            deviation_pct = abs(self.data['close'] - self.data[ma_col]) / self.data[ma_col]
            
            # Overbought: Price significantly above MA
            self.signals[f'price_overextended_above_{ma_type}_{ma_period}'] = (
                (self.data['close'] > self.data[ma_col]) & 
                (deviation_pct > deviation)
            )
            
            # Oversold: Price significantly below MA
            self.signals[f'price_overextended_below_{ma_type}_{ma_period}'] = (
                (self.data['close'] < self.data[ma_col]) & 
                (deviation_pct > deviation)
            )
        
        return self.signals[[f'price_overextended_above_{ma_type}_{ma_period}',
                           f'price_overextended_below_{ma_type}_{ma_period}']]
    
    # ==================== COMPREHENSIVE SIGNAL GENERATION ====================
    
    def generate_all_signals(self):
        """Generate all available SMA/EMA signals"""
        print("GENERATING SMA/EMA TRADING SIGNALS...")
        
        # Crossover signals
        self.golden_death_cross()
        self.ema_crossover(20, 50)
        self.ema_crossover(10, 20)  # Additional fast crossover
        
        # Trend hierarchy
        self.trend_hierarchy([10, 20, 50])
        
        # Bounce signals
        self.ma_bounce_signals(20, 'ema')
        self.ma_bounce_signals(50, 'sma')
        
        # Slope signals
        self.ma_slope_signals(20, 'ema')
        self.ma_slope_signals(50, 'sma')
        
        # Extension signals
        self.price_extension_signals(20, 'ema', 0.02)
        
        # Clean up: Replace NaN values with False
        self.signals = self.signals.fillna(False)
        
        print(f"Generated {len(self.signals.columns)} trading signals")
        return self.signals
    
    def get_signal_summary(self):
        """Get summary of current active signals"""
        if self.signals.empty:
            return "No signals generated yet. Call generate_all_signals() first."
        
        summary = {}
        for col in self.signals.columns:
            active_signals = self.signals[col].sum()
            if active_signals > 0:
                summary[col] = active_signals
        
        return summary

# ==================== USAGE EXAMPLE ====================

# First, create your indicators
trend_calculator = ForexTrendIndicators(df)
df_with_indicators = trend_calculator.add_sma([50, 200])
df_with_indicators = trend_calculator.add_ema([10, 20, 50])

# Then generate signals
signal_gen = SMASignalGenerator(df_with_indicators)
all_signals = signal_gen.generate_all_signals()

# See signal summary
print(signal_gen.get_signal_summary())

# Combine signals with original data
final_df = pd.concat([df_with_indicators, all_signals], axis=1)