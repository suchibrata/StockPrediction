import pandas as pd

def add_default_indicators(df,o='Open',c='Close',h='High',l='Low'):
    df=bollinger_bands(df=df,c=c)
    df=ichimoku(df=df,h=h,l=l)
    df=macd(df=df,c=c)
    df=rsi(df=df,c=c)
    df=stoch(df=df,c=c,l=l,h=h)
    df=adx(df=df,c=c,l=l,h=h)

    return df

def bollinger_bands(df,n=20,c='Close'):
    # bollinger band
    sma = df[c].rolling(window=n).mean()
    rstd = df[c].rolling(window=n).std()
    df['bol_ub']=sma + 2 * rstd
    df['bol_mb'] = sma
    df['bol_lb'] = sma - 2 * rstd

    print("Added bollinger bands to dataframe")
    return df

def ichimoku(df,h='High',l='Low'):
    # ichimoku
    pd9_high = df[h].rolling(window=9).max()
    pd9_low = df[l].rolling(window=9).min()
    df['ichi_tl']=(pd9_high + pd9_low) / 2

    pd26_high = df[h].rolling(window=26).max()
    pd26_low = df[l].rolling(window=26).min()
    df['ichi_sl'] = (pd26_high + pd26_low) / 2
    df['ichi_ls1'] = ((df['ichi_tl'] + df['ichi_sl']) / 2).shift(26)

    pd52_high = df[h].rolling(window=26).max()
    pd52_low = df[l].rolling(window=52).min()
    df['ichi_ls2']= ((pd52_high + pd52_low) / 2).shift(26)

    print("Added ichimoku to dataframe")
    return df

def macd(df,n_fast=12,n_slow=26,n_avg=9,c='Close'):
    exp_1 = df[c].ewm(span=n_fast, adjust=False).mean()
    exp_2 = df[c].ewm(span=n_slow, adjust=False).mean()
    macd_i = exp_1 - exp_2
    exp_3 = macd_i.ewm(span=n_avg, adjust=False).mean()
    df['macd']=macd_i
    df['macd_avg']=exp_3
    print("Added macd to dataframe")
    return df

def rsi(df,n=20,c='Close'):
    delta = df[c].diff()
    d_up, d_down = delta.copy(), delta.copy()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0
    rol_up = d_up.rolling(n).mean()
    rol_down = d_down.rolling(n).mean().abs()
    rs = rol_up / rol_down
    df['rsi'] = 100 - (100 / (1 + rs))

    print("Added rsi to dataframe")
    return df

def stoch(df,n_fast=14,n_slow=3,c='Close',l='Low',h='High'):
    low_min = df[l].rolling(window=n_fast).min()
    high_max = df[h].rolling(window=n_fast).max()
    k_fast = 100 * (df[c] - low_min) / (high_max - low_min)
    k_slow = k_fast.rolling(window=n_slow).mean()
    d_slow = k_slow.rolling(window=n_fast).mean()

    df['stoch_k_fast']=k_fast
    df['stoch_k_slow']=k_slow
    df['stoch_d_slow']=d_slow

    print("Added stoch to dataframe")
    return df

def adx(df, n=14, n_adx=20,c='Close',l='Low',h='High'):
    i = 0
    up_i = []
    do_i = []
    while i + 1 <= df.index[-1]:
        up_move = df.loc[i + 1, h] - df.loc[i, h]
        do_move = df.loc[i, l] - df.loc[i + 1, l]
        if up_move > do_move and up_move > 0:
            up_d = up_move
        else:
            up_d = 0
        up_i.append(up_d)
        if do_move > up_move and do_move > 0:
            do_d = do_move
        else:
            do_d = 0
        do_i.append(do_d)
        i = i + 1
    i = 0
    tr_l = [0]
    while i < df.index[-1]:
        tr = max(df.loc[i + 1, h], df.loc[i, c]) - min(df.loc[i + 1, l], df.loc[i, c])
        tr_l.append(tr)
        i = i + 1
    tr_s = pd.Series(tr_l)
    atr = pd.Series(tr_s.ewm(span=n, min_periods=n).mean())
    up_i = pd.Series(up_i)
    do_i = pd.Series(do_i)
    pos_di = pd.Series(up_i.ewm(span=n, min_periods=n).mean() / atr)
    neg_di = pd.Series(do_i.ewm(span=n, min_periods=n).mean() / atr)
    adx = pd.Series((abs(pos_di - neg_di) / (pos_di + neg_di)).ewm(span=n_adx, min_periods=n_adx).mean(),
                    name='adx')
    df = df.join(adx)

    print("Added adx to dataframe")
    return df









