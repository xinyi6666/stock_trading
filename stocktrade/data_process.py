#!/usr/bin/env python
# coding: utf-8

# In[233]:


import numpy as np
import pandas as pd

path = '~/Downloads/historical_price/'
list_path = '/Users/xinyiluo/Downloads/historical_price/list.txt'
f = open(list_path, 'r')
train_data = []

start_min = (9 + 5) * 60 + 30
end_min = (16 + 5) * 60
early_end_min = (13 + 5) * 60

n_trading_day_2018 = 81
n_trading_day_2019 = 252
n_company = 86

total_minutes = (n_trading_day_2018 + n_trading_day_2019 - 5) * (end_min - start_min + 1) + 5 * (
        early_end_min - start_min + 1)
total_minutes = (252 - 3) * (end_min - start_min + 1) + 3 * (early_end_min - start_min + 1)
# total_minutes = (81 - 2) * (end_min - start_min + 1) + 2 * (early_end_min - start_min + 1)
holidays = [20180101, 20180115, 20180219, 20180330, 20180528, 20180704, 20180903, 20181122,
            20181225, 20190101, 20190121, 20190218, 20190419, 20190527, 20190704, 20190902,
            20191128, 20191225, 20200101, 20200120, 20200217, 20200410,
            20200525, 20200703, 20200907, 20201126, 20201225, 20210101, 20210118, 20210215]

shortened_trading_day = [20181123, 20181224, 20190703, 20191129, 20191224]
VW_IDX = 2
DATE_IDX = 11
MIN_IDX = 9
trading_table_defined = 0


def create_table(trading_days):
    trading_data_30 = np.zeros((total_minutes, n_company + 2))
    counter = 0
    for i in trading_days:
        if i in shortened_trading_day:
            # print(i)
            for j in np.arange(start_min, early_end_min + 1):
                trading_data_30[counter][0] = i
                trading_data_30[counter][1] = j
                counter += 1
        else:
            for j in np.arange(start_min, end_min + 1):
                trading_data_30[counter][0] = i
                trading_data_30[counter][1] = j
                counter += 1

    assert counter == len(trading_data_30)
    return trading_data_30


def add_missing_data(vw_values, date_values, min_values, trading_data_30, idx):
    counter = 0
    i = -1
    while i < len(trading_data) - 2:
        i += 1
        if date_values[i + 1] > date_values[i]:
            if min_values[i] == end_min and date_values[i] not in shortened_trading_day:
                trading_data_30[counter][idx] = vw_values[i]
                counter += 1
                continue

            elif min_values[i] > early_end_min and date_values[i] not in shortened_trading_day:
                if min_values[i] != end_min:
                    gap = end_min - min_values[i] + 1
                    for _ in np.arange(0, gap):
                        trading_data_30[counter][idx] = vw_values[i]
                        counter += 1
                    continue
            elif date_values[i] in shortened_trading_day:
                if min_values[i] < early_end_min:
                    gap = early_end_min - min_values[i] + 1
                    for _ in np.arange(0, gap):
                        trading_data_30[counter][idx] = vw_values[i]
                        counter += 1
                    continue
                if min_values[i] > early_end_min:
                    gap = early_end_min - trading_data_30[counter][1]
                    assert gap >= 0
                    for j in range(int(gap)):
                        trading_data_30[counter][idx] = trading_data_30[counter - 1][idx]
                        counter += 1
                    continue

                if min_values[i] == early_end_min:
                    trading_data_30[counter][idx] = vw_values[i]
                    counter += 1
                    continue

        if date_values[i + 1] == date_values[i]:
            if (date_values[i] in shortened_trading_day) and min_values[i + 1] > early_end_min >= min_values[i]:
                gap = early_end_min - min_values[i] + 1
                for _ in np.arange(0, gap):
                    trading_data_30[counter][idx] = vw_values[i]
                    counter += 1
                i += 1
                j = 1
                while date_values[j + i] == date_values[i] and min_values[i + j] > early_end_min:
                    i += 1

                continue

            if date_values[i] > date_values[i - 1] or i == 0:
                if min_values[i] != start_min:
                    gap = min_values[i] - start_min + 1
                    for _ in np.arange(0, gap):
                        trading_data_30[counter][idx] = vw_values[i]
                        counter += 1
                    if min_values[i + 1] > min_values[i] + 1:
                        gap = min_values[i + 1] - min_values[i]
                        for _ in np.arange(0, gap - 1):
                            trading_data_30[counter][idx] = vw_values[i]
                            counter += 1
                    continue
                else:
                    if min_values[i + 1] > min_values[i] + 1:
                        gap = min_values[i + 1] - min_values[i]
                        for _ in np.arange(0, gap):
                            trading_data_30[counter][idx] = vw_values[i]
                            counter += 1
                    else:
                        trading_data_30[counter][idx] = vw_values[i]
                        counter += 1
                    continue

            if min_values[i + 1] > min_values[i] + 1:
                gap = min_values[i + 1] - min_values[i]
                for _ in np.arange(0, gap):
                    trading_data_30[counter][idx] = vw_values[i]
                    counter += 1
            else:
                if trading_data_30[counter][1].astype(int) != min_values[i]:
                    print(trading_data_30[counter][1])
                    print(min_values[i])
                    print(date_values[i + 1])
                    print(date_values[i])
                    print(counter)
                    break

                trading_data_30[counter][idx] = vw_values[i]
                counter += 1

    if end_min > min_values[len(trading_data) - 1]:
        gap = end_min - min_values[len(trading_data) - 1] + 1
        for _ in range(gap):
            trading_data_30[counter][idx] = vw_values[len(trading_data) - 1]
            counter += 1
    else:
        trading_data_30[counter][idx] = vw_values[len(trading_data) - 1]
        counter += 1

    assert counter == len(trading_data_30)


idx = 2
company_names = ''
for x in f:
    company_names = company_names + ',' + x[:-34]
    print(x[:-34])
    df = pd.read_csv(path + x[:-1])
    df['minute'] = pd.DatetimeIndex(df['t']).minute + pd.DatetimeIndex(df['t']).hour * 60
    df['minute'] = df['minute'].astype(int)
    df['weekday'] = pd.DatetimeIndex(df['t']).weekday
    df['date'] = pd.DatetimeIndex(df['t']).date
    df['date'] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
    year_data = df[df['weekday'].isin([0, 1, 2, 3, 4])]  # select weekday data
    year_data = year_data[~year_data['date'].isin(holidays)]
    year_data = year_data[(year_data['date'] > 20181231) & (year_data['date'] < 20200101)]
    trading_data = year_data[year_data['minute'].isin(np.arange(start_min, end_min + 1, 1))]
    trading_days = trading_data['date'].value_counts();
    trading_days = trading_days.index.to_list()
    trading_days.sort()
    trading_data.to_numpy()
    vw_values = trading_data['vw'].to_numpy()
    date_values = trading_data['date'].to_numpy()
    min_values = trading_data['minute'].to_numpy()
    if not trading_table_defined:
        trading_data_30 = create_table(trading_days)
        trading_table_defined = 1
    add_missing_data(vw_values, date_values, min_values, trading_data_30, idx)
    idx += 1


out_csv = open('./clean_data.csv','a')
title = "data,min"+company_names
out_csv.write("%s\n"%title)
for i in range(len(trading_data_30)):
    print(i)
    out_csv.write("%i,"%trading_data_30[i][0])
    out_csv.write("%i,"%trading_data_30[i][1])
    for j in range(n_company-1):
        out_csv.write("%f,"%trading_data_30[i][2+j])
    out_csv.write("%f\n"%trading_data_30[i][2+n_company-1])

out_csv.close()