#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:37:49 2020

@author: angelo
"""

import csv
import os
from math import sin

class ElectrodePositions():
    def __init__(self,filename):
        self.coordDict = {}
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
        
            for row in reader:
                self.coordDict[row[0]]=[float(row[1]),float(row[2])]
                
    def getCoord(self,channel):
        try:
            return self.coordDict[channel]
        except KeyError:
            print('Channel '+channel+' does not exist in this setup.')
            return False
    
                    
