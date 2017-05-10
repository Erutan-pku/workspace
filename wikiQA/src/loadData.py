# coding=utf-8
# -*- coding: UTF-8 -*- 
import codecs
import pandas as pd
from itertools import groupby

def loadData(dataType) :
    assert dataType in ['train', 'dev', 'test']
    rawData = pd.read_table('../data/WikiQA-%s.tsv'%(dataType), encoding='utf-8')
    #print len(rawData)

    keys = ['QuestionID','SentenceID', 'Sentence', 'Label']
    Datas = [{key:rawData[key][i] for key in keys} for i in range(len(rawData))]
    quMap = {rawData['QuestionID'][i]:rawData['Question'][i] for i in range(len(rawData))}
    
    Datas = groupby(Datas, key=lambda x : x['QuestionID'])
    Datas = [{'QuestionID':key, 'Question':quMap[key], 'sens':[x for x in list(value)]} for key, value in Datas]
    
    [sen.pop('QuestionID') for data in Datas for sen in data['sens']]

    return Datas

def getLabeledData(Datas) :
    return [data for data in Datas if sum([int(sen['Label']) for sen in data['sens']])]

def cleanRawData(Datas, dataType) :
    inputFile  = '../data/WikiQA-%s.tsv'%(dataType)
    outputFile= '../data/WikiQA-%s-clean.tsv'%(dataType)
    QIDs = set([data['QuestionID'] for data in Datas])

    inputs = codecs.open(inputFile, encoding='utf-8').readlines()
    outputs = codecs.open(outputFile, "w", "utf-8")
    outputs.write(inputs[0])
    for line in inputs[1:] :
        if not line.split('\t')[0] in QIDs :
            continue
        outputs.write(line)
    outputs.flush()
    outputs.close()

def writeRef(Datas, dataType) :
    """input full data"""
    # 生成官方给的.ref文件作为评价的标准答案供官方的评价脚本使用。
    outputFile  = '../data/WikiQA-%s.ref'%(dataType)
    outputs = codecs.open(outputFile, "w", "utf-8")
    for i, data in enumerate(Datas) :
        for j, sen in enumerate(data['sens']) :
            outputs.write('%d 0 %d %s\n'%(i+1, j, sen['Label']))
    outputs.flush()
    outputs.close()

def writeRawResults(Datas, filepath, truthLabel='Label') :
    """input full data"""
    # 按照官方模式生成结果文件，作为评价的标准答案供官方的评价脚本使用。
    outputs = codecs.open(filepath, "w", "utf-8")
    for i, data in enumerate(Datas) :
        for j, sen in enumerate(data['sens']) :
            outputs.write('%d 0 %d 0 %.8f 0\n'%(i+1, j, sen[truthLabel]-max(abs(sen[truthLabel]), j)*1e-6))
    outputs.flush()
    outputs.close()

def writeResults(Datas, filepath, truthLabel='Label') :
    output = codecs.open(filepath, "w", "utf-8")
    for data in Datas :
        ID_rank = [sen['SentenceID'] for sen in data['sens']]
        sort_list = [(sen[truthLabel]-max(abs(sen[truthLabel]), i)*1e-6, sen['SentenceID'])for i, sen in enumerate(data['sens'])]
        sort_list = sorted(sort_list, reverse=True)
        sortedHash = {sort_t[1]:i+1 for i, sort_t in enumerate(sort_list)}

        for sen_id in ID_rank :
            output.write('%s\t%s\t%d\n'%(data['QuestionID'], sen_id, sortedHash[sen_id]))
    output.flush()
    output.close()

if __name__ == '__main__':
    trn_data = loadData('train')
    dev_data = loadData('dev')
    tst_data = loadData('test')
    print len(trn_data) # 2117
    print len(dev_data) # 296
    print len(tst_data) # 630

    """
    trn_data = getLabeledData(trn_data)
    dev_data = getLabeledData(dev_data)
    tst_data = getLabeledData(tst_data)
    print len(trn_data) # 872
    print len(dev_data) # 126
    print len(tst_data) # 241

    cleanRawData(trn_data, 'train')
    cleanRawData(dev_data, 'dev')
    cleanRawData(tst_data, 'test')
    #"""

    writeRef(trn_data, 'train')
    writeRef(dev_data, 'dev')
    writeRef(tst_data, 'test')

    writeRawResults(trn_data, '../result/trn_truth_raw')
    writeRawResults(dev_data, '../result/dev_truth_raw')
    writeRawResults(tst_data, '../result/tst_truth_raw')
    
    """
    writeResults(trn_data, 'trn_truth')
    writeResults(dev_data, 'dev_truth')
    writeResults(tst_data, 'tst_truth')
    #"""
