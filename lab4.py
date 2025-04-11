import numpy as np
import pandas as pd
import timeit


def read_with_pandas():
    file_path = r'D:\kpi\AD\lab4\household_power_consumption.txt'
    data = pd.read_csv(file_path, sep=';', decimal='.', na_values='?')

    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
    data.dropna(inplace=True)
    
    return data


def read_with_numpy():
    file_path = r'D:\kpi\AD\lab4\household_power_consumption.txt'
    column_types = [
        ("Date", "U10"), ("Time", "U8"),
        ("Global_active_power", "f8"), ("Global_reactive_power", "f8"),
        ("Voltage", "f8"), ("Global_intensity", "f8"),
        ("Sub_metering_1", "f8"), ("Sub_metering_2", "f8"), ("Sub_metering_3", "f8")
    ]
    arr = np.genfromtxt(file_path, delimiter=';', dtype=column_types, skip_header=1, encoding='utf-8', missing_values='?')

    valid = ~np.isnan(arr['Global_active_power']) & ~np.isnan(arr['Voltage']) & ~np.isnan(arr['Global_intensity'])
    arr = arr[valid]
    hour_values = np.array([int(t.split(':')[0]) for t in arr['Time']])
    
    return arr, hour_values


def evaluate_tasks(pandas_data, numpy_data, hour_array):
    print("Перше завдання:\n")

    def np_filter_high_power(data):
        return data[data['Global_active_power'] > 5]

    def pd_filter_high_power(df):
        return df[df['Global_active_power'] > 5]

    def np_voltage_check(data):
        return data[data['Voltage'] > 235]

    def pd_voltage_check(df):
        return df[df['Voltage'] > 235]

    def np_combined_filter(data):
        condition = (data['Global_intensity'] >= 19) & (data['Global_intensity'] <= 20)
        compare = data['Sub_metering_1'] + data['Sub_metering_2'] > data['Sub_metering_3']
        return data[condition & compare]

    def pd_combined_filter(df):
        mask = (df['Global_intensity'].between(19, 20)) & (
            df['Sub_metering_1'] + df['Sub_metering_2'] > df['Sub_metering_3']
        )
        return df[mask]

    def np_avg_submetering(data):
        sample = np.random.choice(data, size=500000, replace=False)
        values = np.vstack((sample['Sub_metering_1'], sample['Sub_metering_2'], sample['Sub_metering_3']))
        return np.mean(values, axis=1)

    def pd_avg_submetering(df):
        subset = df.sample(500000, random_state=123)
        return subset[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()

    def np_evening_filter(data, hours):
        evening = (hours > 18) & (data['Global_active_power'] > 6)
        filtered = data[evening]

        sm = np.vstack((filtered['Sub_metering_1'], filtered['Sub_metering_2'], filtered['Sub_metering_3'])).T
        dominant = np.argmax(sm, axis=1)
        selected = filtered[dominant == 1]

        middle = len(selected) // 2
        combined = np.concatenate((selected[:middle:3], selected[middle::4]))

        return combined

    def pd_evening_filter(df):
        subset = df[(df['datetime'].dt.hour > 18) & (df['Global_active_power'] > 6)]
        dominant = subset[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].idxmax(axis=1)
        subset = subset[dominant == 'Sub_metering_2']

        first = subset.iloc[:len(subset)//2:3]
        second = subset.iloc[len(subset)//2::4]
        return pd.concat([first, second])

    tasks_list = [
        ("> 5 кВт", np_filter_high_power, pd_filter_high_power),
        ("> 235 В", np_voltage_check, pd_voltage_check),
        ("Складна фільтрація", np_combined_filter, pd_combined_filter),
    ]

    for desc, np_func, pd_func in tasks_list:
        np_time = timeit.timeit(lambda: np_func(numpy_data), number=10)
        pd_time = timeit.timeit(lambda: pd_func(pandas_data), number=10)
        print(f"Завдання {desc}:")
        print(f"NumPy: {np_time:.4f} с")
        print(f"Pandas: {pd_time:.4f} с")
        print("Швидший:", "NumPy" if np_time < pd_time else "Pandas", "\n")

    np_t4 = timeit.timeit(lambda: np_avg_submetering(numpy_data), number=10)
    pd_t4 = timeit.timeit(lambda: pd_avg_submetering(pandas_data), number=10)

    print("Середні значення Sub_metering (Завдання 4):")
    print(f"NumPy час: {np_t4:.4f} с")
    print(f"Pandas час: {pd_t4:.4f} с")
    np_avg = np_avg_submetering(numpy_data)
    pd_avg = pd_avg_submetering(pandas_data)
    print(f"NumPy: SM1={np_avg[0]:.2f}, SM2={np_avg[1]:.2f}, SM3={np_avg[2]:.2f}")
    print(f"Pandas: SM1={pd_avg['Sub_metering_1']:.2f}, SM2={pd_avg['Sub_metering_2']:.2f}, SM3={pd_avg['Sub_metering_3']:.2f}\n")

    np_t5 = timeit.timeit(lambda: np_evening_filter(numpy_data, hour_array), number=10)
    pd_t5 = timeit.timeit(lambda: pd_evening_filter(pandas_data), number=10)
    print("Завдання 5: Кількість записів після фільтрації:")
    print(f"NumPy час: {np_t5:.4f} с, Результат: {len(np_evening_filter(numpy_data, hour_array))}")
    print(f"Pandas час: {pd_t5:.4f} с, Результат: {len(pd_evening_filter(pandas_data))}")


def run():
    df = read_with_pandas()
    arr, hours = read_with_numpy()
    evaluate_tasks(df, arr, hours)


if __name__ == '__main__':
    run()
