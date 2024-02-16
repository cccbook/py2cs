import requests
import time
import sqlite3
import os

TWSE_URL = 'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json'
CSV_FILE_NAME = "stockThisMonth.csv"

def get_web_content(stock_id, current_date):
    resp = requests.get(TWSE_URL + '&date=' + current_date + '&stockNo=' + stock_id)
    if resp.status_code != 200:
        return None
    else:
        return resp.json()


def get_data(stock_id, current_date):
    info = list()
    resp = get_web_content(stock_id, current_date)
    if resp is None:
        return None
    else:
        if resp['data']:
            for data in resp['data']:
                record = {
                    '日期': data[0],
                    '開盤價': data[3],
                    '收盤價': data[6],
                    '成交筆數': data[8]
                }
                info.append(record)
        return info

def execute_sql(db, sql_cmd):
    cursor = db.cursor()
    cursor.execute(sql_cmd)
    db.commit()

STOCK_DB_FILE = 'stockPrice.db'
def main():
    stock_id = '2330'
    current_date = time.strftime('%Y%m%d')
    current_year = time.strftime('%Y')
    current_month = time.strftime('%m')
    print('取得本月台積電 (2330) 的股價 %s %s...' % (current_year, current_month))
    get_data(stock_id, current_date)
    collected_info = get_data(stock_id, current_date)
    # 寫入 sqlite
    if os.path.isfile(STOCK_DB_FILE):
        os.remove(STOCK_DB_FILE)
    db = sqlite3.connect(STOCK_DB_FILE)
    execute_sql(db, f'CREATE TABLE stockPrice (日期 TEXT, 開盤價 INTEGER, 收盤價 INTEGER, 成交筆數 INTEGER)')
    for info in collected_info:
        print(info)
        command = f'INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("{info["日期"]}", {info["開盤價"]}, {info["收盤價"]}, {info["成交筆數"].replace(",", "")})'
        print('command=', command)
        execute_sql(db, command)
    db.close()

if __name__ == '__main__':
    main()
