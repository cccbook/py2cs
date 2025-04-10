from PIL import Image

img = Image.open( "../img/test.png" )
img.thumbnail( (400,100) ) #指定長與寬並進行縮圖製作
img.save( "test_thumbnail.jpg" )
print (img.size)