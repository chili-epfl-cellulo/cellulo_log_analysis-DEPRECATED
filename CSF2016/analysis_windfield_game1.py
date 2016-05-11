import numpy as np
import sys
import pickle
import datetime
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime, date, time, timedelta
import ast
from itertools import izip_longest
from scipy.misc import imread
import matplotlib.cbook as cbook
import math
import matplotlib as mplt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from numpy import genfromtxt
import seaborn as sns
from scipy.stats import kendalltau
sns.set(color_codes=True)

infile = "windfield_game1_blue_withindex.log"
count = 0
width = 1280
height = 1280
global hlist
hlist = []
hp =0
mapwidth = 1700
mapheight = 660
nbcols = int(2418 /12)
nbrows = int(950 /12)
fps = 0
global lines
lines = []
global img
maxd = math.sqrt(nbcols*nbcols + nbrows*nbrows)

marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='gray')



def plotPlayground():
    img =  imread('./playground.jpg')
    plt.imshow(img, zorder = 0, extent=[0.0, nbcols, 0.0, nbrows])
    plt.show()


def readLog(fname='logfile', delim=','):
    f = open(fname, 'r')
    lines = f.readlines()
    return lines


def computeFrameRate(lines):
    firstline = lines[0].split(' ')
    lastline = lines[-1].split(' ')
    if(firstline[0] == lastline[0]): # data from the same day
        first_time = datetime.strptime(firstline[1], '%H:%M:%S.%f')
        last_time = datetime.strptime(lastline[1], '%H:%M:%S.%f')
    return len(lines)/(last_time - first_time).total_seconds()

def addIndexBis(lines,fname=infile[:-4]+'_withindex.log'):
    count = 0
    fout = open(fname,'w')
    prev_time = datetime.strptime(lines[0].split(' ')[1], '%H:%M:%S')
    current_time = datetime.strptime(lines[0].split(' ')[1], '%H:%M:%S')
    for line in lines:
        current_time = datetime.strptime(line.split(' ')[1], '%H:%M:%S')
        if(current_time>prev_time):
            prev_time = current_time
            current_time = line.split(' ')[1]
            count = 0
        if(line.find("robot at")>0):
            count+=1
        line = line[0:19]+'.'+ "{0:0=2d}".format(count) + line[19:]
        fout.write(line)
    fout.close()

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def computeDistancePPoints(hlist, robotposition):
    distancelist = []
    if(len(hlist)>0):
        for hpoint in hlist :
            hx = float(hpoint[0])/nbrows
            hy = float(hpoint[1])/nbcols
            if(hpoint[3]=='true'):
                d = 0
            else:
                d = distance([hx,hy],robotposition)
            if(d>maxd):
                print('pb distance ', hx, hy, robotposition)
            distancelist.append(d)
        return distancelist
    else:
        return [-1]

def extractData(lines,getPosition=True,getVel=True,getPpint=True,getEvents=True):
    plt.figure()
    img =  imread('./playground.jpg')
    hlist=[]
    phlist =[]
    robot_position = [(0,0)]
    robot_velocity = []
    ppoints=[]
    events = []
    distances = []
    maxlenght =0
    fout = open('./csv/'+infile[:-4]+'_distances.csv','w')
    fgrasped = open('./csv/'+infile[:-4]+'_grasped.csv','w')
    fkidnapped = open('./csv/'+infile[:-4]+'_kidnapped.csv','w')
    fkidnappedDistance = open('./csv/'+infile[:-4]+'_kidnapped_distances.csv','w')
    fvel = open('./csv/'+infile[:-4]+'_velocity.csv','w')
    fpos = open('./csv/'+infile[:-4]+'_position.csv','w')
    fpoint = open('./csv/'+infile[:-4]+'_ppoints.csv','w')
    firstline = lines[0].split(' ')
    first_time = datetime.strptime(firstline[0] + ' '+firstline[1], '%d/%m/%Y %H:%M:%S.%f')
    time = 0
    dx = 0
    dy = 0
    for line in lines:

        linelist = line.split(' ')
        time = linelist[0] + ' '+ linelist[1]
        #time = time.total_seconds()
        #print(time)

        # position of the robot
        if(line.find("robot at") >= 0 and getPosition):
            robot_positionx = float(linelist[-2])
            robot_positiony = float(linelist[-1])
            if(robot_positionx>1 or robot_positiony>1):
                continue
            robot_position.append(( robot_positionx, robot_positiony))
            all_distances = computeDistancePPoints(hlist,[robot_positionx,robot_positiony])

            distances.append([time, len(hlist), all_distances ])
            maxlenght = max(maxlenght,len(hlist))

            fout.write((time+','+str(len(all_distances))+','+str(all_distances).replace(']','').replace('[','')+'\n'))
            fpos.write((time+','+str(robot_positionx)+','+str(robot_positiony)+'\n'))
            #print(distances[-1])

        # when a new ppoint list is hidden or re-hidden
        if(line.find('hide ppoint list') >= 0 and getPpint):
            tmplist =linelist[-1]
            hlist = tmplist[1:-1].split(',')
            hlist = parsePointList(hlist)
            if(hlist!=phlist):
                phlist=hlist
                print('new')
            print(phlist,hlist)
            #print(set(hlist).difference(set(phlist)))
            for h in hlist:
                fpoint.write((time+','+str(h).replace(']','').replace('[','')+'\n'))
                ppoints.append([time, h])

        # speed of robot
        if(line.find('robot velocity')>=0 and getVel):
            robot_velx = int(float(linelist[-2]))
            robot_vely = int(float(linelist[-1]))
            robot_velocity.append([time, robot_velx, robot_vely])
            fvel.write((time+','+str(robot_velx)+','+str(robot_vely)+'\n'))

        # events
        if(line.find('grasped')>=0 and getEvents):
            fgrasped.write(time+',grasped\n')
            events.append([time,'GRASPED'])
        if(line.find('checking')>=0 and getEvents):
            events.append([time,'CHECKING'])

        #kidnappes
        if(line.find('returned on paper')>=0):
            fkidnapped.write(str(time)+',returned\n')
            (dxb, dyb) = robot_position[-1]
            dx = dxb - dx
            dy = dyb - dy
            fkidnappedDistance.write(str(time)+','+str(dx)+','+str(dy)+'\n')
            events.append([time,'RETURNED'])
        elif(line.find('was kidnapped')>=0):
            (dx,dy) = robot_position[-1]

            fkidnapped.write(str(time)+',kidnapped\n')
            events.append([time,'KIDNAPPED'])

    fout.close()
    fgrasped.write(str(time)+',over\n')
    fgrasped.close()
    fkidnapped.write(str(time)+',over\n')
    fkidnapped.close()
    fkidnappedDistance.close()
    fvel.close()
    fpos.close()
    fpoint.close()
    print(len(robot_position), len(ppoints),len(robot_velocity),len(events))
    print(maxlenght)
    return (robot_position, ppoints, robot_velocity, distances)

def parsePointList(hlist):
    hpointlist = []
    for h in xrange(0,len(hlist),4):
        hx = hlist[h]
        hy = hlist[h+1]
        hw = hlist[h+2]
        hv = hlist[h+3]
        hpointlist.append([int(hx),int(hy),hw,hv])
    return hpointlist

def drawHiddenPointList(hlist,img):
    for h in xrange(0,len(hlist),4):
        hx = hlist[h]
        hy = hlist[h+1]
        hw = hlist[h+2]
        hv = hlist[h+3]
        if(hv == 'true'):
            hxm = int(hy)
            hym = int(hx)
            c = ('blue'  if hw == '-3' else 'red')
            plt.plot(hxm,hym, fillstyle='full', marker='o',color=c, zorder=1)
            #fill(c[0],c[1],c[2],127)
            #ellipse(hxm, hym, 70, 70)

def velocityAnalysis(color='orange'):
    fname = "./csv/windfield_game1_"+color+"_withindex_velocity.csv"
    v_data = genfromtxt(fname,delimiter=',',dtype=object,missing_values=0, skip_header = 1,
                converters={0: lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S.%f'),
                1: np.float, 2: np.float},usecols=(1,2))

    v_data = v_data.view(np.float64).reshape(v_data.shape + (-1,))

    x = [row[0] for row in v_data]
    y = [row[1] for row in v_data]

    sns.set(style="white", color_codes=True)
    g = sns.jointplot(x=np.array(x)*nbcols, y=np.array(y)*nbrows)
    plt.show()

def computeVelocityfromPosition(color='orange'):
    color = ['orange','blue','green']
    sns.set_style('darkgrid')
    x = np.empty(1)
    y = np.empty(1)
    vx = np.empty(1)
    vy = np.empty(1)
    for c in color:
        fname = "./csv/windfield_game1_"+c+"_withindex_position.csv"
        v_data = genfromtxt(fname,delimiter=',',dtype=object,missing_values=0, skip_header = 1,
                converters={0: lambda xl: datetime.strptime(xl, '%d/%m/%Y %H:%M:%S.%f'),
                1: np.float, 2: np.float},usecols=(1,2))
        v_data = v_data.view(np.float64).reshape(v_data.shape + (-1,))
        x = np.append(x, np.array([row[0] for row in v_data]))
        y = np.append(y,np.array([row[1] for row in v_data]))
        vx = np.diff(x)
        vy = np.diff(y)

    #sns.jointplot(np.array(x), np.array(y), kind="hex", stat_func=kendalltau, color="#4CB391")
    sns.set(style="white", color_codes=True)
    sns.jointplot(x=np.array(vx)*mapwidth, y=np.array(vy)*mapheight,kind="kde", space=0)
    #img =  imread('./playground.jpg')

    #plt.imshow(img, zorder = 0, extent=[0.0, mapwidth, 0.0, mapheight])
    #g.savefig('positionOccupation.svg', format='png',dpi=1200)
    #plt.show()

def extractNbTrialTillFind(color='orange'):
    fname = "./csv/windfield_game1_"+color+"_withindex_ppoints.csv"
    v_data = genfromtxt(fname,delimiter=',',dtype=object,missing_values=0, skip_header = 1,
                converters={0: lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S.%f'),
                1: np.int, 2: np.int,3:np.str,4:np.str},usecols=range(0,5))
    np.array(v_data)
    hlist = np.empty((250,250),dtype=datetime)
    ppointlisttime = []
    for t, x, y, h,v in v_data:
        if(hlist[x,y]==None and v.find('true')>=0):
            hlist[x,y]=t
        elif(hlist[x,y]!=None and v.find('false')>=0):
            #trouve
            print(hlist[x,y])
            ppointlisttime.append((t - hlist[x,y]).total_seconds())
            hlist[x,y]==0
    print(ppointlisttime)

def plotDistanceEvol(color='orange'):
    fname = "./csv/windfield_game1_"+color+"_withindex_ppoints.csv"

def positionAnalysis():
    color = ['orange','blue','green']
    sns.set_style('darkgrid')
    x = np.empty(1)
    y = np.empty(1)
    for c in color:
        fname = "./csv/windfield_game1_"+c+"_withindex_position.csv"
        v_data = genfromtxt(fname,delimiter=',',dtype=object,missing_values=0, skip_header = 1,
                converters={0: lambda xl: datetime.strptime(xl, '%d/%m/%Y %H:%M:%S.%f'),
                1: np.float, 2: np.float},usecols=(1,2))
        v_data = v_data.view(np.float64).reshape(v_data.shape + (-1,))
        x = np.append(x, np.array([row[0] for row in v_data]))
        y = np.append(y,np.array([row[1] for row in v_data]))
        print(len(x),len(y))
    #sns.jointplot(np.array(x), np.array(y), kind="hex", stat_func=kendalltau, color="#4CB391")
    sns.set(style="white", color_codes=True)
    g = sns.jointplot(x=np.array(x)*mapwidth, y=np.array(y)*mapheight,kind="kde", space=0)
    img =  imread('./playground.jpg')

    plt.imshow(img, zorder = 0, extent=[0.0, mapwidth, 0.0, mapheight])
    #g.savefig('positionOccupation.svg', format='png',dpi=1200)
    plt.show()




def graspAnalysis(color = 'orange'):
    th_time = timedelta(seconds = 2)
    g_data = readLog( "./csv/windfield_game1_"+color+"_withindex_grasped.csv")
    tot_gtime =0
    first_time = timeOf(g_data[0].split(',')[0])
    gtime = timeOf(g_data[0].replace('\n','').split(',')[0])
    fgrasp = open('./csv/grasped_periods.csv','a')
    fgrasp.write('tablet,time,grasped_time\n')
    for gline in g_data:
        gline = gline.replace('\n','').split(',')
        if((timeOf(gline[0]) < gtime + th_time) and gline[1]!='over'):
            gtime = timeOf(gline[0])
            continue
        elif(gline[1]=='over'):
            tot_gtime /= (timeOf(gline[0]) - first_time).total_seconds()
            fgrasp.close()
            print(tot_gtime*100, gline[0])
            return tot_gtime*100
        else:
            dt =  timeOf(gline[0])
            if((dt - gtime).total_seconds()<1000):
                tot_gtime += (dt - gtime).total_seconds()
                fgrasp.write(color+','+gline[0]+','+str((dt - gtime).total_seconds())+'\n')
            gtime = dt

def timeOf(astring):
    return datetime.strptime(astring,'%d/%m/%Y %H:%M:%S.%f')

def kidnappedRatio(color = 'orange'):
    k_data = readLog( "./csv/windfield_game1_"+color+"_withindex_kidnapped.csv")
    tot_ktime = 0
    ktime = 0
    prev_lineisk = False
    fkidnapped = open('./csv/kidnapped_periods.csv','a')
    fkidnapped.write('tablet,time,kidnapped_time\n')
    first_time = timeOf(k_data[0].split(',')[0])
    for kline in k_data:
        kline = kline.replace('\n','').split(',')
        if(kline[1]=='kidnapped'):
            ktime = timeOf(kline[0])
            prev_lineisk = True
        elif(kline[1]=='returned' and prev_lineisk):
            dt =  timeOf(kline[0])
            tot_ktime += (dt - ktime).total_seconds()
            fkidnapped.write(color+','+kline[0]+','+str((dt - ktime).total_seconds())+'\n')
            prev_lineisk = False
        elif(kline[1]=='over'):
            tot_ktime /= (timeOf(kline[0]) - first_time).total_seconds()
            prev_lineisk = False
            print(tot_ktime*100, kline[0])
            fkidnapped.close()
            return tot_ktime*100


def occupationAnalysis():
    img = plt.imread("playground.jpg")
    robot_position = readLog( "./csv/windfield_game1_green_withindex_position.csv")
    data=np.zeros((nbcols, nbrows))
    for robotp in robot_position:
        robotp = robotp.split(',')
        print(robotp[0])
        px = int(float(robotp[1]) * nbcols)
        py = int(float(robotp[2]) * nbrows)
        data[px][py]+=1

    robot_position = readLog( "./csv/windfield_game1_orange_withindex_position.csv")
    for robotp in robot_position:
        robotp = robotp.split(',')
        print(robotp[0])
        px = int(float(robotp[1]) * nbcols)
        py = int(float(robotp[2]) * nbrows)
        data[px][py]+=1

    robot_position = readLog( "./csv/windfield_game1_blue_withindex_position.csv")
    for robotp in robot_position:
        robotp = robotp.split(',')
        print(robotp[0])
        px = int(float(robotp[1]) * nbcols)
        py = int(float(robotp[2]) * nbrows)
        data[px][py]+=1
    fig, ax = plt.subplots()
    #heatmap = ax.pcolor(data)
    red_high = ((0., 0., 0.),
         (.3, .5, 0.5),
         (1., 1., 1.))

    blue_middle = ((0., .2, .2),
         (.3, .5, .5),
         (.8, .2, .2),
         (1., .1, .1))

    green_none = ((0,0,0),(1,0,0))

    cdict3 = {'red':  red_high,
     'green': green_none,
     'blue': blue_middle,
     'alpha': ((0.0, 0.0, 0.0),
               (0.3, 0.5, 0.5),
               (1.0, 1.0, 1.0))
    }

    #ax.scatter(x, y, label=str(i), color=color, alpha=0.5)
    #dropout_high = LinearSegmentedColormap('Dropout', cdict3)
    #plt.register_cmap(cmap = dropout_high)
    sns.jointplot(x="x", y="y", data=data, kind="kde");
    #x,y = data.T
    #plt.hexbin(x,y, bins='log', cmap=dropout_high)
    #plt.imshow(img,zorder=0, extent=[0, nbcols, 0, nbrows])
    #plt.show()
    #plt.imshow(data)
    #plt.show()


def main():
    #mplt.use('Agg')
    lines  = readLog(infile)
    (robot_position, ppoints, robot_velocity, distances) =  extractData(lines)

    #print(computeFrameRate(lines))

    #occupationAnalysis()

    #print((kidnappedRatio('orange')+kidnappedRatio('green')+kidnappedRatio('blue'))/3)

    #print((graspAnalysis('orange')+graspAnalysis('green')+graspAnalysis('blue'))/3)

    #graspAnalysis(color = 'orange')

    #velocityAnalysis(color='green')

    #positionAnalysis()

    #computeVelocityfromPosition()

    #extractNbTrialTillFind(color='orange')
    #import seaborn as sns; sns.set(style="white", color_codes=True)
    #tips = sns.load_dataset("tips")
    #g = sns.jointplot(x="total_bill", y="tip", data=tips)
    #plt.show()

def plotDistances(distances):
    times = [d[0] for d in distances]
    distancelist = [d[-1] for d in distances]
    pt1 = [d[0] for d in distancelist]
    pt2 = [d[1] for d in distancelist]
    pt3 = [d[2] for d in distancelist]
    pt4 = [d[3] for d in distancelist]
    print(pt1)
    plt.plot(pt1)
    plt.show()





if __name__ == '__main__':
    main()
