# coding=utf-8
# -*- coding: UTF-8 -*- 
"""
参考：
http://blog.csdn.net/u012464190/article/details/24914317
"""
import sys
import os  
def countFileLines(filename):  
    handle = open(filename,'rb').readlines() 
    return len(handle)
  
global count_n
count_n = 0
def listdir(dirPath):  
    filePaths = [os.path.join(dirPath, aFile) for aFile in os.listdir(dirPath) \
        if not aFile[0] == '.']
    
    
    dirCounts = [listdir(path) for path in filePaths if os.path.isdir(path)]
    fileCounts = [countFileLines(path) for path in filePaths \
        if all([not os.path.isdir(path), path.endswith('.py')])]
    count_lines = sum(dirCounts) + sum(fileCounts)
    return count_lines

if __name__ == '__main__':
    dirPath = 'NLPCC_KBQA/' 

    count = listdir(dirPath)  
    print count