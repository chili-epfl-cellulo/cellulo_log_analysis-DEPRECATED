"""
 * Create Image. 
 * 
 * The createImage() function provides a fresh buffer of pixels to play with.
 * This example creates an image gradient.
 """

from itertools import izip_longest


infile = "../windfield_game1_green.log"
count = 0
width = 1280
height = 1280
global hlist
hlist = []
hp =0
def settings():
    size(640 * 2, 250 * 2 +100)
    
def setup():
    global f
    f = open(infile)
    global img
    img = loadImage("../playground.jpg")
    pixCount = len(img.pixels)
    global robot_position
    robot_position = (0, 0)
    img.resize(img.width * 2, img.height * 2)

def draw():
    background(0)
    image(img, 0, 0)
    line = f.readline()
    global hlist
    global robot_position
    linelist = line.split(' ')
    if(line.find("robot at") >= 0):
        robot_positionx = int(float(linelist[-2]) * img.width)
        robot_positiony = int(float(linelist[-1]) * img.height)
        robot_position = [robot_positionx, robot_positiony]
    fill(0,255,0,127)
    ellipse(robot_position[0], robot_position[1], 10, 10)
    if(line.find('hide ppoint list') >= 0):
        hlist = line.split(' ')[-1]
        #hp = hlist.split(',').lenght/4
    if(len(hlist)>0):
        drawHiddenPointList(hlist)
        
    textSize(32)
    fill(255,255,255,255)
    text(linelist[0] + ' ' + linelist[1], 30, img.height + 50)
    #img.pixels[robot_position.x*img.width, robot.position.y *img.height] = color(255,0,0)


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def drawHiddenPointList(hlist):
    for hx, hy, hw, hv in grouper(4, hlist):
        if(hv == 'true'):
            hx = int(hx) * img.width / 12
            hy = int(hy) * img.height / 12
            c = ((0, 0, 255, 127)  if hw == '-3' else (255, 0, 0, 127))
            fill(c)
            print(hx, hy)
            ellipse(hx, hy, 10, 10)
