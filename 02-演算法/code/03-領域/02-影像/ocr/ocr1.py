from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

print(pytesseract.image_to_string('eng1.png', lang="eng"))

# 中文 OCR 要先下載訓練檔
# -- https://github.com/tesseract-ocr/tessdata_best/blob/main/chi_tra.traineddata
# 下載完畢後放到 C:\Program Files\Tesseract-OCR\tessdata 才能執行下列這行
print(pytesseract.image_to_string('tw1.png', lang="chi_tra"))
