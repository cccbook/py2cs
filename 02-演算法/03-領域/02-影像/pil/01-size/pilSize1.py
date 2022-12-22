from PIL import Image
img1 = Image.open('../img/test.png')
width, height = img1.size
max1 = max(width, height)
size = 200
quartersizedIm = img1.resize((int(size*width/max1), int(size*height/max1)))
quartersizedIm.save('testSmall.png')
