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

infile = "windfield_game1_blue_withindex.log"
count = 0
width = 1280
height = 1280
global hlist
hlist = []
hp =0
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


def extractData(lines):
    plt.figure()
    img =  imread('./playground.jpg')
    hlist=[]
    robot_position = []
    robot_velocity = []
    ppoints=[]
    events = []
    distances = []
    maxlenght =0
    fout = open('./csv/'+infile[:-4]+'_distances.csv','w')
    fgrasped = open('./csv/'+infile[:-4]+'_grasped.csv','w')
    fvel = open('./csv/'+infile[:-4]+'_velocity.csv','w')
    fpos = open('./csv/'+infile[:-4]+'_position.csv','w')
    fpoint = open('./csv/'+infile[:-4]+'_ppoints.csv','w')
    firstline = lines[0].split(' ')
    first_time = datetime.strptime(firstline[0] + ' '+firstline[1], '%d/%m/%Y %H:%M:%S.%f')

    for line in lines:

        linelist = line.split(' ')
        time = datetime.strptime(linelist[0] + ' '+ linelist[1], '%d/%m/%Y %H:%M:%S.%f') - first_time
        time = time.total_seconds()
        #print(time)

        # position of the robot
        if(line.find("robot at") >= 0):
            robot_positionx = float(linelist[-2])
            robot_positiony = float(linelist[-1])
            if(robot_positionx>1 or robot_positiony>1):
                continue
            robot_position.append([time, robot_positionx, robot_positiony])
            all_distances = computeDistancePPoints(hlist,[robot_positionx,robot_positiony])

            distances.append([time, len(hlist), all_distances ])
            maxlenght = max(maxlenght,len(hlist))
            fout.write((str(time)+','+str(all_distances).replace(']','').replace('[','')+'\n'))
            fpos.write((str(time)+','+str(robot_positionx)+','+str(robot_positiony)+'\n'))
            #print(distances[-1])

        # when a new ppoint list is hidden or re-hidden
        if(line.find('hide ppoint list') >= 0):
            tmplist =linelist[-1]
            hlist = tmplist[1:-1].split(',')
            hlist = parsePointList(hlist)
            fpoint.write((str(time)+','+str(hlist).replace(']','').replace('[','')+'\n'))
            ppoints.append([time, hlist])

        # speed of robot
        if(line.find('robot velocity')>=0):
            robot_velx = int(float(linelist[-2]))
            robot_vely = int(float(linelist[-1]))
            robot_velocity.append([time, robot_velx, robot_vely])
            fvel.write((str(time)+','+str(robot_velx)+','+str(robot_vely)+'\n'))

        # events
        if(line.find('grasped')>=0):
            fgrasped.write(str(time)+',grasped\n')
            events.append([time,'GRASPED'])
        if(line.find('checking')>=0):
            events.append([time,'CHECKING'])

    fout.close()
    fgrasped.close()
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

def graspAnalysis(fname = './csv/'+infile[:-4]+'_distances.csv'):
    lines = readLog(fname)
    prev_time = 0
    data = []
    





def main():
    lines  = readLog(infile)
    #print(computeFrameRate(lines))
    (robot_position, ppoints, robot_velocity, distances) =  extractData(lines)
    #plotDistances(distances)
    #fname2 = infile[:-4]+'_distances.csv'
    #plt.plotfile(fname2, cols=(0,1,2,3), subplots = False)
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
