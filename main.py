import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

def process_historical_data(file_path='full_data.csv'):
    """Завантажує дані, очищає їх та заповнює дні без тривог нулями."""
    print("Завантаження та очищення даних...")
    df = pd.read_csv(file_path)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['date'] = df['started_at'].dt.date
    
    daily_alerts = df.groupby('date').size().reset_index(name='alert_count')
    daily_alerts['date'] = pd.to_datetime(daily_alerts['date'])
    daily_alerts.set_index('date', inplace=True)
    
    # Заповнення пропущених днів нулями для безперервного часового ряду
    full_date_range = pd.date_range(start=daily_alerts.index.min(), end=daily_alerts.index.max(), freq='D')
    daily_alerts = daily_alerts.reindex(full_date_range, fill_value=0)
    
    return daily_alerts

def run_arima_forecast(daily_alerts, steps=7):
    """Будує прогноз за допомогою моделі ARIMA."""
    print(f"Побудова ARIMA прогнозу на {steps} днів...")
    # order=(7,1,1) для врахування тижневої сезонності
    model = ARIMA(daily_alerts['alert_count'], order=(7, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def plot_gantt_timeline(forecast_csv='full_weekly_forecast_7_days.csv'):
    """Візуалізує таймлайн тривог по областях (Діаграма Ганта)."""
    print("Малюємо таймлайн...")
    try:
        df = pd.read_csv(forecast_csv)
    except FileNotFoundError:
        print(f"Файл {forecast_csv} не знайдено. Перевір наявність бази прогнозу.")
        return

    # Фільтруємо лише підтверджені загрози
    df = df[df['Чи буде тривога?'] == 'Так'].copy()

    def parse_start_time(row):
        date_str = row['Дата']
        time_str = row['Час початку']
        start_hour = int(time_str.split(':')[0])
        return pd.to_datetime(date_str) + timedelta(hours=start_hour)

    df['start_datetime'] = df.apply(parse_start_time, axis=1)
    df['duration_min'] = pd.to_numeric(df['Тривалість (хв)'], errors='coerce').fillna(60)
    df['end_datetime'] = df['start_datetime'] + pd.to_timedelta(df['duration_min'], unit='m')

    threat_colors = {
        'Обстріл / Швидка балістика': '#e74c3c',
        'Балістика (Тилові)': '#e67e22',
        'Крилаті ракети / МіГ': '#3498db',
        'Ударні БПЛА / Довгий МіГ': '#9b59b6',
        'Комбінована атака': '#2c3e50'
    }
    df['color'] = df['Тип загрози'].map(threat_colors).fillna('#95a5a6')

    fig, ax = plt.subplots(figsize=(18, 12))
    oblasts = sorted(df['Область'].unique(), reverse=True)
    obl_to_y = {obl: i for i, obl in enumerate(oblasts)}

    for idx, row in df.iterrows():
        y = obl_to_y[row['Область']]
        start = mdates.date2num(row['start_datetime'])
        end = mdates.date2num(row['end_datetime'])
        ax.barh(y, width=end-start, left=start, height=0.5, color=row['color'], edgecolor='black', alpha=0.8)

    ax.set_yticks(range(len(oblasts)))
    ax.set_yticklabels(oblasts, fontsize=10)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%Y'))
    plt.xticks(rotation=0, fontsize=10)
    
    plt.title('Таймлайн прогнозованих повітряних тривог (Горизонт: 1 тиждень)', fontsize=16, pad=20)
    plt.xlabel('Дата та Час', fontsize=12)
    plt.grid(axis='x', which='both', linestyle='--', alpha=0.5)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    
    handles = [mpatches.Patch(color=color, label=label) for label, color in threat_colors.items()]
    plt.legend(handles=handles, title='Тип загрози', loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig('timeline_forecast.png', dpi=300)
    print("Готово! Графік збережено як timeline_forecast.png")

if __name__ == "__main__":
    # Щоб запустити весь пайплайн, розкоментуй ці рядки, коли матимеш full_data.csv:
    # daily_data = process_historical_data('full_data.csv')
    # my_forecast = run_arima_forecast(daily_data, steps=7)
    
    # Візуалізація зведеного тижневого прогнозу
    plot_gantt_timeline('full_weekly_forecast_7_days.csv')