from PIL import Image
import sys
img = Image.open(sys.argv[1])
img_mod = Image.open(sys.argv[2])
size = width , height = img.size
img_ans = Image.new("RGBA", (width,height))
for x in range(width):
    for y in range(height):
        if list( img.getpixel( (x,y) ) ) != list( img_mod.getpixel( (x,y) ) ):
            img_ans.putpixel( (x,y) , img_mod.getpixel( (x,y) ) )
        else:
            img_ans.putpixel((x,y),(0,0,0,0))
img_ans.save('ans_two.png')
