"""
 * Create Image. 
 * 
 * The createImage() function provides a fresh buffer of pixels to play with.
 * This example creates an image gradient.
 """
infile = "windfield_game1_blue.log"
count = 0
def setup():
    size(640, 360)
    important = []
    globale line
    f = open(infile)
    line = f.readline()

    for line in f:
        for phrase in keep_phrases:
            if phrase in line:
                important.append(line)
                break



    global img
    img = createImage(230, 230, ARGB)
    pixCount = len(img.pixels)
    for i in range(pixCount):
        a = map(i, 0, pixCount, 255, 0)
        img.pixels[i] = color(0, 153, 204, a)
        
    


def draw():
    line = f.readline()
    
    background(0)
    image(img, 90, 80)
    image(img, mouseX - img.width / 2, mouseY - img.height / 2)
