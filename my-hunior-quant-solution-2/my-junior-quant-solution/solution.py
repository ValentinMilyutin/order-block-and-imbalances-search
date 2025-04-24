import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta


# Reading the data
df = pd.read_csv('D:/Python_and_Jupyter_notebooks/trade_algorithm/my-hunior-quant-solution-2/my-junior-quant-solution/market-data.csv', header=None)

# Setting the columns and types
df.columns = ['datetime_str', 'open', 'high', 'low', 'close', 'volume']
df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
df.drop(columns=['datetime_str'], inplace=True)
df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
df[numeric_cols] = df[numeric_cols].astype(float)
df = df.sort_values('datetime').reset_index(drop=True)
# print(df.head())

# Resampling to 1H intervals
df.set_index('datetime', inplace=True)
df_h = df.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
df_h = df_h.reset_index()
# print(df_h.head())

print(df_h)

# Detecting local highs and lows
df_h['is_local_high'] = (df_h['high'].shift(1) < df_h['high']) & (df_h['high'] > df_h['high'].shift(-1))
df_h['is_local_low']  = (df_h['low'].shift(1) > df_h['low']) & (df_h['low'] < df_h['low'].shift(-1))
# print(df_h.head())

# Detecting order blocks
body_min_i   = df_h[['open', 'close']].min(axis=1)
body_max_i   = df_h[['open', 'close']].max(axis=1)
body_min_ip1 = df_h[['open', 'close']].shift(-1).min(axis=1)
body_max_ip1 = df_h[['open', 'close']].shift(-1).max(axis=1)
# Engulfing condition: body of i+1 overlaps body of i
# is_engulfing = (body_min_ip1 < body_min_i) & (body_max_ip1 > body_max_i) # Строгое перекрытие телом
# is_engulfing = (body_max_i - body_min_i).abs() < (body_max_ip1 - body_min_ip1).abs() # Сравнение по размеру тела
tolerance = 0.05 * (body_max_i - body_min_i)
is_engulfing = (
    (body_min_ip1 <= body_min_i + tolerance) &
    (body_max_ip1 >= body_max_i - tolerance)
)
df_h['is_bearish_order_block'] = (df_h['is_local_high'] & is_engulfing).astype(bool)
df_h['is_bullish_order_block'] = (df_h['is_local_low']  & is_engulfing).astype(bool)
# print(df_h[['local_high', 'local_low', 'bearish_order_block', 'bullish_order_block']].head())

# Detecting 1h Fair Value Gaps
df_h['fvg_type'] = None
df_h['fvg_size'] = 0.0
high_1 = df_h['high'].shift(0)
low_1 = df_h['low'].shift(0)
high_3 = df_h['high'].shift(-2)
low_3 = df_h['low'].shift(-2)
bullish_fvg = low_3 > high_1
bullish_gap = low_3 - high_1
bearish_fvg = high_3 < low_1
bearish_gap = low_1 - high_3
df_h.loc[bullish_fvg, 'fvg_type'] = 'bullish'
df_h.loc[bullish_fvg, 'fvg_size'] = bullish_gap[bullish_fvg]
df_h.loc[bearish_fvg, 'fvg_type'] = 'bearish'
df_h.loc[bearish_fvg, 'fvg_size'] = bearish_gap[bearish_fvg]
# looking at results
df_h['ob_type'] = None
df_h.loc[df_h['is_bearish_order_block'] == True, 'ob_type'] = 'bearish'
df_h.loc[df_h['is_bullish_order_block'] == True, 'ob_type'] = 'bullish'
df_ob = df_h[df_h['ob_type'].notna()][['datetime', 'ob_type', 'fvg_type', 'fvg_size']]
# print(df_ob)

# Calculating the range of the order block
# Prepare space for OB range
df_h['ob_low'] = np.nan
df_h['ob_high'] = np.nan
# Get shifted values for next 2 candles (c2 = pullback, c3 = confirmation)
low_c2 = df_h['low'].shift(-1)
high_c2 = df_h['high'].shift(-1)
# ----- Rule 1: FVG present → use full candle (with wicks) -----
bullish_fvg = (df_h['is_bullish_order_block']) & (df_h['fvg_type'] == 'bullish')
bearish_fvg = (df_h['is_bearish_order_block']) & (df_h['fvg_type'] == 'bearish')
df_h.loc[bullish_fvg, 'ob_low'] = df_h.loc[bullish_fvg, 'low']
df_h.loc[bullish_fvg, 'ob_high'] = df_h.loc[bullish_fvg, 'high']
df_h.loc[bearish_fvg, 'ob_low'] = df_h.loc[bearish_fvg, 'low']
df_h.loc[bearish_fvg, 'ob_high'] = df_h.loc[bearish_fvg, 'high']
# ----- Rule 2: No FVG → use body + directional wick -----
bullish_no_fvg = (df_h['is_bullish_order_block']) & (df_h['fvg_type'].isna())
bearish_no_fvg = (df_h['is_bearish_order_block']) & (df_h['fvg_type'].isna())
# For bullish: low = min(open, close, wick low), high = max(open, close)
df_h.loc[bullish_no_fvg, 'ob_low'] = df_h.loc[bullish_no_fvg, ['open', 'close', 'low']].min(axis=1)
df_h.loc[bullish_no_fvg, 'ob_high'] = df_h.loc[bullish_no_fvg, ['open', 'close']].max(axis=1)
# For bearish: high = max(open, close, wick high), low = min(open, close)
df_h.loc[bearish_no_fvg, 'ob_high'] = df_h.loc[bearish_no_fvg, ['open', 'close', 'high']].max(axis=1)
df_h.loc[bearish_no_fvg, 'ob_low'] = df_h.loc[bearish_no_fvg, ['open', 'close']].min(axis=1)
# ----- Rule 3: Adjust OB range -----
# Expand bullish OB high if c2 has a lower wick
mask_bull_expand = (df_h['is_bullish_order_block']) & (low_c2 < df_h['low'])
df_h.loc[mask_bull_expand, 'ob_high'] = df_h['high']
# Expand bearish OB low if c2 has a higher wick
mask_bear_expand = (df_h['is_bearish_order_block']) & (high_c2 > df_h['high'])
df_h.loc[mask_bear_expand, 'ob_low'] = df_h['low']

with pd.option_context('display.max_columns', None, 'display.width', None):
    print(df_h.loc[df_ob.index, ['datetime', 'open', 'high', 'low', 'close', 'ob_type' , 'ob_low', 'ob_high']])

def candle_stick_ob(df, title, block_width=timedelta(hours=1)):
    """
    Plots a high-resolution candlestick chart with annotated order blocks and saves it as an image.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing OHLC (open, high, low, close) data along with order block annotations.
        Required columns:
            - 'datetime' : Timestamps for each candlestick (datetime64)
            - 'open', 'high', 'low', 'close' : Price data for candlestick plotting (float)
            - 'ob_type' : Type of order block ('bullish' or 'bearish') or NaN (str or NaN)
            - 'ob_low', 'ob_high' : Vertical bounds for the order block rectangles (float)

    title : str
        The title of the chart. Also used to name the output image file.

    block_width : timedelta, optional
        The horizontal width of each order block (default is 1 hour).

    Returns
    -------
    None
        Saves a 1920x1080 PNG image with doubled resolution to the current directory.
    """

    fig = go.Figure(data=go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='lime',
        decreasing_line_color='red'
    ))

    for _, row in df.iterrows():
        if pd.notna(row['ob_type']):
            fill_color = 'rgba(0, 255, 0, 0.6)' if row['ob_type'] == 'bullish' else 'rgba(255, 0, 0, 0.6)'
            border_color = 'lime' if row['ob_type'] == 'bullish' else 'red'

            fig.add_shape(
                type='rect',
                x0=row['datetime'],
                x1=row['datetime'] + block_width,
                y0=row['ob_low'],
                y1=row['ob_high'],
                xref='x',
                yref='y',
                line=dict(color=border_color, width=1),
                fillcolor=fill_color,
                opacity=0.6,
                layer='below'
            )

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        width=1920,
        height=1080,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=14),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.write_image(f"D:/Python_and_Jupyter_notebooks/trade_algorithm/my-hunior-quant-solution-2/my-junior-quant-solution/{title.replace(' ', '-')}.png", width=1920, height=1080, scale=2)



candle_stick_ob(df_h, '1H Candlestick Chart with Order Blocks')

## Нахождение имбалансов (FVG) на интервале 15 минут
# Resampling at 15 min intervals
df_15m = df.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna().reset_index()

df_15m['fvg_type'] = None
df_15m['fvg_low'] = np.nan
df_15m['fvg_high'] = np.nan

# Get current candle and 3rd candle ahead
high_1 = df_15m['high']
low_1 = df_15m['low']
high_3 = df_15m['high'].shift(-2)
low_3 = df_15m['low'].shift(-2)

# Bullish FVG: gap up
mask_bullish = low_3 > high_1
df_15m.loc[mask_bullish, 'fvg_type'] = 'bullish'
df_15m.loc[mask_bullish, 'fvg_low'] = high_1[mask_bullish]
df_15m.loc[mask_bullish, 'fvg_high'] = low_3[mask_bullish]

# Bearish FVG: gap down
mask_bearish = high_3 < low_1
df_15m.loc[mask_bearish, 'fvg_type'] = 'bearish'
df_15m.loc[mask_bearish, 'fvg_low'] = high_3[mask_bearish]
df_15m.loc[mask_bearish, 'fvg_high'] = low_1[mask_bearish]

def plot_imbalances(df_15m, title, gap_width=timedelta(minutes=30)):
    """
    Plots 15m candlestick chart with:
    - Fair Value Gaps (FVGs) labeled and shaded
    - Gray candles (light = bullish, dark = bearish) labeled in the legend
    - Ultra HD image export (3840x2160)

    Parameters
    ----------
    df_15m : pd.DataFrame
        Must include:
        ['datetime', 'open', 'high', 'low', 'close', 'fvg_type', 'fvg_low', 'fvg_high']

    title : str
        Title and filename.

    gap_width : timedelta
        Horizontal span of FVG zones (default: 30min).

    Returns
    -------
    None
    """
    fig = go.Figure()

    # Actual price candles (dull gray)
    fig.add_trace(go.Candlestick(
        x=df_15m['datetime'],
        open=df_15m['open'],
        high=df_15m['high'],
        low=df_15m['low'],
        close=df_15m['close'],
        name='Price',
        showlegend=False,
        increasing_line_color='rgba(180,180,180,0.5)',  # light gray
        decreasing_line_color='rgba(80,80,80,0.5)'      # dark gray
    ))

    # FVG zones
    for _, row in df_15m.iterrows():
        if row['fvg_type'] == 'bullish':
            fig.add_shape(
                type='rect',
                x0=row['datetime'],
                x1=row['datetime'] + gap_width,
                y0=row['fvg_low'],
                y1=row['fvg_high'],
                xref='x', yref='y',
                fillcolor='rgba(0, 255, 0, 0.4)',
                line=dict(color='lime', width=1),
                layer='below'
            )
        elif row['fvg_type'] == 'bearish':
            fig.add_shape(
                type='rect',
                x0=row['datetime'],
                x1=row['datetime'] + gap_width,
                y0=row['fvg_low'],
                y1=row['fvg_high'],
                xref='x', yref='y',
                fillcolor='rgba(255, 0, 0, 0.4)',
                line=dict(color='red', width=1),
                layer='below'
            )

    # Dummy traces for labels
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=20, color='rgba(180,180,180,0.7)'),
        name='Bullish Candle (light gray)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=20, color='rgba(80,80,80,0.7)'),
        name='Bearish Candle (dark gray)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=20, color='rgba(0,255,0,0.7)'),
        name='Bullish FVG'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=20, color='rgba(255,0,0,0.7)'),
        name='Bearish FVG'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        width=3840,
        height=2160,
        font=dict(color='white', size=18),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5,
            font=dict(size=20),
            itemwidth=100,
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=80, r=80, t=120, b=80)
    )

    # Save ultra-HD image
    fig.write_image(f"D:/Python_and_Jupyter_notebooks/trade_algorithm/my-hunior-quant-solution-2/my-junior-quant-solution/{title.replace(' ', '-')}.png", width=3840, height=2160, scale=4)


plot_imbalances(df_15m, '15-Minute Imbalance Zones')

## Этап 2
# 1. Create shifted time windows from df_h
start_times = df_h['datetime'] + pd.Timedelta(hours=1)
end_times = df_h['datetime'] + pd.Timedelta(hours=2)

# 2. Broadcast comparison (vectorized)
# Outer comparison: df_15m['datetime'][:, None] vs start_times[None, :]
time_15m = df_15m['datetime'].values[:, None]  # shape (N, 1)
start_arr = start_times.values[None, :]        # shape (1, M)
end_arr = end_times.values[None, :]            # shape (1, M)

# 3. Create a boolean mask: which df_15m times fall into which df_h windows
mask = (time_15m >= start_arr) & (time_15m < end_arr)

# 4. For each 15m row, get the index of the matching df_h row (if any)
match_idx = mask.argmax(axis=1)
no_match = ~mask.any(axis=1)

# Optional: tag with df_h index or a column (e.g., order block type)
df_15m['matched_df_h_idx'] = np.where(no_match, np.nan, match_idx)

# Now optionally join additional columns from df_h
df_15m['matched_ob_type'] = np.where(
    no_match,
    np.nan,
    df_h['ob_type'].values[match_idx]
)

cols_q = ['datetime', 'fvg_low', 'fvg_high', 'fvg_type', 'matched_df_h_idx']

# print(df_15m.query("fvg_type == matched_ob_type")[cols_q])

# Сначала фильтруем только те, у кого есть соответствие по индексу и типу
df_15m_filtered = df_15m[
    df_15m['matched_df_h_idx'].notna() &
    (df_15m['fvg_type'] == df_15m['matched_ob_type'])
].copy()

# Преобразуем индекс в int
df_15m_filtered['matched_df_h_idx'] = df_15m_filtered['matched_df_h_idx'].astype(int)

# Подтягиваем границы OB
df_15m_filtered['ob_low'] = df_15m_filtered['matched_df_h_idx'].map(df_h['ob_low'])
df_15m_filtered['ob_high'] = df_15m_filtered['matched_df_h_idx'].map(df_h['ob_high'])

# Проверка на вхождение FVG в OB
price_mask = (
    (df_15m_filtered['fvg_low'] >= df_15m_filtered['ob_low']) &
    (df_15m_filtered['fvg_high'] <= df_15m_filtered['ob_high'])
)

# Финальный датафрейм
df_15m_target = df_15m_filtered[price_mask][cols_q + ['ob_low', 'ob_high']].copy()

# Объединяем с существующими условиями
df_15m_filtered = df_15m.query("fvg_type == matched_ob_type").copy()
df_15m_filtered = df_15m_filtered.loc[price_mask]

# Сопоставим тип FVG и OB
df_15m['type_match'] = df_15m['fvg_type'] == df_15m['matched_ob_type']
df_15m_matched = df_15m[df_15m['type_match'] & df_15m['matched_df_h_idx'].notna()].copy()

# Получаем ценовые границы OB
df_15m_matched['matched_df_h_idx'] = df_15m_matched['matched_df_h_idx'].astype(int)
df_15m_matched['ob_low'] = df_15m_matched['matched_df_h_idx'].map(df_h['ob_low'])
df_15m_matched['ob_high'] = df_15m_matched['matched_df_h_idx'].map(df_h['ob_high'])

# Проверка: входит ли FVG в ценовой диапазон OB
price_mask = (
    (df_15m_matched['fvg_low'] >= df_15m_matched['ob_low']) &
    (df_15m_matched['fvg_high'] <= df_15m_matched['ob_high'])
)

# Итоговый датафрейм
df_15m_target = df_15m_matched[price_mask][cols_q + ['ob_low', 'ob_high']].copy()

# Convert matched_df_h_idx to int
df_15m_target['matched_df_h_idx'] = df_15m_target['matched_df_h_idx'].astype(int)

# Assign ob_low and ob_high from df_h using matched indices
df_15m_target['ob_low'] = df_15m_target['matched_df_h_idx'].map(df_h['ob_low'])
df_15m_target['ob_high'] = df_15m_target['matched_df_h_idx'].map(df_h['ob_high'])

df_15m_target['fvg_low_clipped'] = np.maximum(df_15m_target['fvg_low'], df_15m_target['ob_low'])
df_15m_target['fvg_high_clipped'] = np.minimum(df_15m_target['fvg_high'], df_15m_target['ob_high'])

pd.set_option('display.width', 0)
print(df_15m_target[cols_q+['ob_low', 'ob_high', 'fvg_low_clipped', 'fvg_high_clipped']])
pd.reset_option('display.width')

## Формирование итоговой таблицы
# Copy to avoid modifying originals
df_h_display = df_h.loc[df_ob.index,].copy()
df_15m_display = df_15m_target.copy()

# Format datetime as required (OBs are 1h + original time)
df_h_display['datetime_fmt'] = (df_h_display['datetime'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M %d.%m.%y')
df_15m_display['datetime_fmt'] = (df_15m_display['datetime'] + pd.Timedelta(minutes=15)).dt.strftime('%H:%M %d.%m.%y')

# Format price ranges
df_h_display['price_range'] = df_h_display['ob_low'].map(lambda x: f"{x:.2f}$") + ' - ' + df_h_display['ob_high'].map(lambda x: f"{x:.2f}$")
df_15m_display['price_range'] = df_15m_display['fvg_low_clipped'].map(lambda x: f"{x:.2f}$") + ' - ' + df_15m_display['fvg_high_clipped'].map(lambda x: f"{x:.2f}$")

pd.set_option('display.width', 0)
# Build formatted order blocks
ob_rows = []
for i, row in enumerate(df_h_display.itertuples(), start=1):
    ob_rows.append({
        'Порядковый номер': f'{i}',
        'Формация (ордер блок или имбаланс)': 'Ордер блок',
        'Направление (тип)': 'Бычий' if row.ob_type == 'bullish' else 'Медвежий',
        'Дата и время формирования': row.datetime_fmt,
        'Диапазон цен': row.price_range
    })

    # Add matching imbalances
    df_imb = df_15m_display[df_15m_display['matched_df_h_idx'] == row.Index]
    for j, imb_row in enumerate(df_imb.itertuples(), start=1):
        ob_rows.append({
            'Порядковый номер': f'{i}.{j}',
            'Формация (ордер блок или имбаланс)': 'Имбаланс',
            'Направление (тип)': 'Бычий' if imb_row.fvg_type == 'bullish' else 'Медвежий',
            'Дата и время формирования': imb_row.datetime_fmt,
            'Диапазон цен': imb_row.price_range
        })

# Final DataFrame
df_formatted = pd.DataFrame(ob_rows)

print(df_formatted)

df_formatted.to_excel('D:/Python_and_Jupyter_notebooks/trade_algorithm/my-hunior-quant-solution-2/my-junior-quant-solution/order_blocks_and_imbalances.xlsx', index=False)