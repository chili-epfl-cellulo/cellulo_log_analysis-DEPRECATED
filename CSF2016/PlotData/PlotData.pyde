"""
 * Create Image. 
 * 
 * The createImage() function provides a fresh buffer of pixels to play with.
 * This example creates an image gradient.
 """

from itertools import izip_longest
import ast

infile = "../windfield_game1_orange.log"
count = 0
width = 1280
height = 1280
global hlist
hlist = []
hp =0
nbcols = int(2418 /12)
nbrows = int(950 /12)



def settings():
    size(640 * 2, 250 * 2 +100)
    
def setup():
    frameRate(120)
    global f
    f = open(infile, 'r')
    global lines
    lines = f.readlines()
    global img
    img = loadImage("../playground.jpg")
    pixCount = len(img.pixels)
    global robot_position
    robot_position = (0, 0)
    img.resize(img.width * 2, img.height * 2)

def draw():
    #print(frameRate)
    background(0)
    image(img, 0, 0)
    global count
    line = lines[count]
    global hlist
    global robot_position
    linelist = line.split(' ')
    if(line.find("robot at") >= 0):
        robot_positionx = int(float(linelist[-2]) * img.width)
        robot_positiony = int(float(linelist[-1]) * img.height)
        robot_position = [robot_positionx, robot_positiony]
    fill(0,255,0,127)
    ellipse(robot_position[0], robot_position[1], 50, 50)
    if(line.find('hide ppoint list') >= 0):
        tmplist =line.split(' ')[-1]
        print(tmplist)
        hlist = tmplist[1:-1].split(',')
        print(hlist)
        #hp = hlist.split(',').lenght/4
    if(len(hlist)>0):
        drawHiddenPointList(hlist)
        
    textSize(32)
    fill(255,255,255,255)
    text(linelist[0] + ' ' + linelist[1], 30, img.height + 50)
    count +=1
    #img.pixels[robot_position.x*img.width, robot.position.y *img.height] = color(255,0,0)


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def drawHiddenPointList(hlist):
    for h in xrange(0,len(hlist),4):
        hx = hlist[h]
        hy = hlist[h+1]
        hw = hlist[h+2]
        hv = hlist[h+3]
        if(hv == 'true'):
            hxm = map(int(hx), 0, nbrows, 0, img.width)
            hym =  map(int(hy), 0, nbcols, 0, img.height) 
            c = ((0, 0, 255, 127)  if hw == '-3' else (255, 0, 0, 127))
            fill(c[0],c[1],c[2],127)
            ellipse(hxm, hym, 70, 70)
