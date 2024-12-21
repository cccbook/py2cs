import openpyxl

# 利用 Workbook 建立一個新的工作簿
workbook = openpyxl.Workbook()

# 取得第一個工作表
sheet = workbook.worksheets[0]

# 設定 sheet 工作表 A1 儲存格內容為 "Hello Python, Hello Excel."
sheet['A1'] = 'Hello Python, Hello Excel.'

# 儲存檔案
workbook.save('test.xlsx')