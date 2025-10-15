import os
import io
import time
import telebot
import requests
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime

# === TELEGRAM TOKEN ===
TELEGRAM_TOKEN = "7977043428:AAHacF1Gw48QvnPOBgi2OIW5zIbG2w6NiCE"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# basit watch liste ileride genişletebiliriz
watchlist = {}

# --------------------- Yardımcılar --------------------- #
def smart_decimals(price: float) -> int:
    """Fiyat büyüklüğüne göre akıllı ondalık basamak sayısı seç."""
    p = abs(float(price))
    if p >= 1000: return 2
    if p >= 1:     return 2
    if p >= 0.1:   return 4
    if p >= 0.01:  return 5
    if p >= 0.001: return 6
    return 8

def fmt(price: float, decimals: int) -> str:
    """Ondalık sayıya göre güvenli biçimleme (negatifleri sıfırın altına düşürmez)."""
    if price is None or np.isnan(price):
        return "-"
    if price < 0:
        price = 0.0
    return f"{price:.{decimals}f}"

def get_data(symbol, interval="1h", period="30d"):
    """Önce Yahoo (yfinance), olmazsa Binance dene."""
    ticker = symbol.upper().replace("/", "-")
    if ticker.endswith("USDT"):
        ticker = ticker.replace("USDT", "USD")
    # yfinance
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is not None and not df.empty:
            return df[['Open','High','Low','Close','Volume']]
    except Exception as e:
        print("[yfinance error]", e)

    # binance
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}"
        res = requests.get(url, timeout=10)
        kl = res.json()
        if isinstance(kl, list) and len(kl) > 0:
            df = pd.DataFrame(kl, columns=[
                'OpenTime','Open','High','Low','Close','Volume',
                'CloseTime','QuoteAssetVolume','Trades','TakerBaseVol','TakerQuoteVol','Ignore'
            ])
            df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
            df.set_index('OpenTime', inplace=True)
            df = df[['Open','High','Low','Close','Volume']].astype(float)
            return df
    except Exception as e:
        print("[binance error]", e)

    return None

def calc_atr(df, n=14):
    """Average True Range"""
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def fib_levels(df):
    """Bilgi amaçlı (hediyelik), TP artık ATR tabanlı."""
    hi = df['High'].max()
    lo = df['Low'].min()
    diff = hi - lo
    return {
        '0.236': hi - diff * 0.236,
        '0.382': hi - diff * 0.382,
        '0.5':   hi - diff * 0.5,
        '0.618': hi - diff * 0.618,
        '0.786': hi - diff * 0.786
    }

# --------------------- Analiz --------------------- #
def analyze(symbol, interval="1h"):
    df = get_data(symbol, interval, "30d")
    if df is None or len(df) < 60:
        return None, f"Hata: Veri yok ya da yetersiz. ({symbol})"

    c = df['Close']
    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    # RSI
    up = c.diff().clip(lower=0)
    dn = -c.diff().clip(upper=0)
    rs = (up.rolling(14).mean() / dn.rolling(14).mean()).replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    # MACD (12-26)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()

    last_price = float(c.iloc[-1])
    d = smart_decimals(last_price)

    last_rsi  = float(rsi.iloc[-1])
    last_macd = float(macd.iloc[-1])
    last_e20  = float(ema20.iloc[-1])
    last_e50  = float(ema50.iloc[-1])

    atr = float(calc_atr(df, 14))
    atr = max(atr, 10**(-d))  # aşırı küçüklerde sıfırlanmayı önle

    # Yön tespiti
    if last_e20 > last_e50 and last_macd > 0:
        trend = "LONG"
    elif last_e20 < last_e50 and last_macd < 0:
        trend = "SHORT"
    else:
        trend = "HOLD"

    # === ATR tabanlı TP ve SL ===
    steps = [0.5, 1.0, 1.5, 2.0, 2.5]

    if trend == "LONG":
        targets = [max(0.0, last_price + k*atr) for k in steps]
        stop = max(0.0, last_price - 1.3*atr)
    elif trend == "SHORT":
        targets = [max(0.0, last_price - k*atr) for k in steps]  # SHORT için aşağı yön
        # yakın->uzak sıralama (fiyata yakın olan önce)
        targets = sorted(targets, reverse=False)
        stop = max(0.0, last_price + 1.3*atr)
    else:
        targets = []
        stop = last_price

    # Trend skoru
    score = 5
    if last_e20 > last_e50: score += 2
    if last_macd > 0:       score += 1
    if 45 < last_rsi < 70:  score += 1
    if atr / last_price < 0.02: score += 1
    score = min(score, 10)

    # Yorum
    yon = "yükseliş" if trend == "LONG" else "düşüş" if trend == "SHORT" else "yatay"
    yorum = f"{symbol.upper()} {yon} trendinde. RSI {last_rsi:.1f} seviyesinde, "
    if last_rsi > 70:
        yorum += "aşırı alım bölgelerinde. "
    elif last_rsi < 30:
        yorum += "aşırı satım bölgelerinde. "
    yorum += f"EMA20 {'>' if last_e20>last_e50 else '<'} EMA50, MACD {'pozitif' if last_macd>0 else 'negatif'}. "
    yorum += "Momentum güçlü, trend destekleniyor." if score > 6 else "Momentum zayıf, kararsız seyir var."

    # Mesaj
    msg =  "📊 " + ("LONG" if trend=="LONG" else "SHORT" if trend=="SHORT" else "HOLD") + " POZİSYONU\n"
    msg += f"💰 COIN: {symbol.upper()}\n\n"
    msg += f"📈 Fiyat: {fmt(last_price, d)}\n"
    msg += f"📊 RSI: {last_rsi:.2f} | MACD: {last_macd:.2f}\n"
    msg += f"📉 EMA20: {fmt(last_e20, d)} | EMA50: {fmt(last_e50, d)}\n"
    msg += f"⚡ Trend Gücü: {score}/10\n\n"
    for i, t in enumerate(targets, 1):
        msg += f"🎯 TP{i}: {fmt(t, d)}\n"
    msg += f"⛔ STOP-LOSS: {fmt(stop, d)}\n\n"
    msg += yorum + "\n"
    msg += "🟢 Para girişi." if trend=="LONG" else "🔴 Para çıkışı." if trend=="SHORT" else "⚪ Nötr hareket."

    return df, msg

# --------------------- Flow (MFI/CMF/RSI + Hacim) --------------------- #
def flow_analysis(symbol):
    df = get_data(symbol, "1h", "14d")
    if df is None or len(df) < 50:
        return f"Hata: {symbol} için veri yok."
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']

    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = 100 - (100 / (1 + (pos / (neg + 1e-9))))

    cmf = (((c - l) - (h - c)) / (h - l + 1e-9) * v).rolling(20).mean()

    up = c.diff().clip(lower=0)
    dn = -c.diff().clip(upper=0)
    rs = (up.rolling(14).mean() / dn.rolling(14).mean()).replace([np.inf,-np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))

    avg_vol = v.rolling(20).mean()
    vol_surge = v.iloc[-1] > avg_vol.iloc[-1] * 1.5
    vol_mult = (v.iloc[-1] / (avg_vol.iloc[-1] + 1e-9))

    msg = f"💧 PARA AKIŞI ANALİZİ ({symbol.upper()})\n\n"
    msg += f"💹 RSI: {rsi.iloc[-1]:.1f} | MFI: {mfi.iloc[-1]:.1f} | CMF: {cmf.iloc[-1]:.3f}\n"
    msg += f"📊 Hacim: {'ARTTI' if vol_surge else 'normal'} ({vol_mult:.2f}x)\n\n"

    if mfi.iloc[-1] > 60 and cmf.iloc[-1] > 0:
        yorum = "Kurumsal ilgisi güçlü, net alım baskısı var."
        durum = "🟢 Güçlü para girişi."
    elif mfi.iloc[-1] < 40 and cmf.iloc[-1] < 0:
        yorum = "Perakende satışı baskın, çıkış eğilimi sürüyor."
        durum = "🔴 Para çıkışı."
    else:
        yorum = "Kararsız yapı; net yön teyit bekliyor."
        durum = "⚪ Nötr hareket."

    return f"{msg}💬 {yorum}\n{durum}"

# --------------------- Telegram Komutları --------------------- #
@bot.message_handler(commands=['start'])
def start_cmd(m):
    bot.reply_to(m,
        "Selam reyiz! Bot aktif 🚀\n\n"
        "Komutlar:\n"
        "/analiz SYMBOL [30m|1h|4h]\n"
        "/flow SYMBOL (para akışı)\n"
        "/watch_add SYMBOL (trend alarmı)\n"
        "/watch_list\n"
        "/watch_remove SYMBOL\n\n"
        "Örnek: /analiz BTCUSDT"
    )

@bot.message_handler(commands=['analiz'])
def analiz_cmd(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.reply_to(m, "Kullanım: /analiz BTCUSDT [1h|4h|30m]")
        return
    symbol = parts[1].upper()
    interval = parts[2] if len(parts) > 2 else "1h"
    bot.reply_to(m, f"[INFO] Veri çekiliyor: {symbol} ({interval})")

    df, text = analyze(symbol, interval)
    if df is None:
        bot.reply_to(m, text)
        return

    # Grafik
    buf = io.BytesIO()
    mpf.plot(df[-120:], type='candle', mav=(20,50), volume=True,
             title=f"{symbol} {interval}", style='yahoo',
             savefig=dict(fname=buf, dpi=120, bbox_inches='tight'))
    buf.seek(0)
    bot.send_photo(m.chat.id, buf, caption=text)

@bot.message_handler(commands=['flow'])
def flow_cmd(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.reply_to(m, "Kullanım: /flow BTCUSDT")
        return
    symbol = parts[1]
    msg = flow_analysis(symbol)
    bot.reply_to(m, msg)

# Watch komutlarını basit placeholder olarak bırakalım (ikinci etapta zamanlayıcı eklenir)
@bot.message_handler(commands=['watch_add'])
def watch_add(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.reply_to(m, "Kullanım: /watch_add BTCUSDT")
        return
    s = parts[1].upper()
    watchlist[s] = True
    bot.reply_to(m, f"{s} takip listesine eklendi. (Alarm motoru ikinci etapta çalışacak)")

@bot.message_handler(commands=['watch_list'])
def watch_list(m):
    if not watchlist:
        bot.reply_to(m, "Takip listesi boş.")
        return
    items = "\n".join([f"• {k}" for k in watchlist.keys()])
    bot.reply_to(m, "Takip Listesi:\n" + items)

@bot.message_handler(commands=['watch_remove'])
def watch_remove(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.reply_to(m, "Kullanım: /watch_remove BTCUSDT")
        return
    s = parts[1].upper()
    if s in watchlist:
        del watchlist[s]
        bot.reply_to(m, f"{s} listeden çıkarıldı.")
    else:
        bot.reply_to(m, f"{s} listede yok.")

# --------------------- Main --------------------- #
if __name__ == "__main__":
    print("Bot çalışıyor reyiz 💹")
    bot.infinity_polling(skip_pending=True)
