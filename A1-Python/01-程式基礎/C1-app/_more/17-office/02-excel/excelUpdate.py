import openpyxl

# 使用 load_workbook 讀取 test.xlsx
workbook = openpyxl.load_workbook('test.xlsx')

# 取得第一個工作表
sheet = workbook.worksheets[0]

# 設定 sheet 工作表 A2 儲存格內容為 "Test Excel"
sheet['A2'] = 'Test Excel.'

# 儲存檔案
workbook.save('test.xlsx')