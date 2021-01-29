import csv
import numpy as np
import os

def accSapa(row):
    acc = row.split('(')[0]
    remain = row.split('(')[1]
    percentage = remain.split('%')[0]

    return float(acc), float(percentage)

def accAssem(num,percentage):
    num = round(num,3)
    percentage = round(percentage,2)
    reStr = str(num)+'('+str(percentage)+'%)'
    return reStr

def accAssemInt(num,percentage):
    num = int(num)
    percentage = round(percentage,2)
    reStr = str(num)+'('+str(percentage)+'%)'
    return reStr

ccsv = []
avg_ccsv = []

os.system("rm ./braodTable.csv")
outfile = open('braodTable.csv', 'a', newline='')
spamwriter = csv.writer(outfile, delimiter=',')
spamwriter.writerow(['model','Dataset','exptNo','TestAcc_S', '(lost%)','ValAcc_S','(lost%)','FLOPs_S','(% of initial)','Paras_S','(% of initial)','TestAcc_F','(lost%)','ValAcc_F','(lost%)','FLOPs_F','(% of initial)','Paras_F','(% of initial)'])


for i in range(4):
    if i == 0:
        files = ['cnnsingle_1004_e1',
                 'cnnsingle_1004_e2',
                 'cnnsingle_1004_e3',
                 'cnnsingle_1005_e1',
                 'cnnsingle_1005_e2',
                 'cnnsingle_1005_e3',
                 'cnnsingle_1006_e1',
                 'cnnsingle_1006_e2',
                 'cnnsingle_1006_e3'
                 ]
        layers = 4
    if i ==1:
        files = ['mlpmulti_1004_e1',
                 'mlpmulti_1004_e2',
                 'mlpmulti_1004_e3',
                 'mlpmulti_1005_e1',
                 'mlpmulti_1005_e2',
                 'mlpmulti_1005_e3',
                 'mlpmulti_1006_e1',
                 'mlpmulti_1006_e2',
                 'mlpmulti_1006_e3'
                 ]
        layers = 4

    if i ==2:
        files = ['cnnmulti_1004_e1',
                 'cnnmulti_1004_e2',
                 'cnnmulti_1004_e3',
                 'cnnmulti_1005_e1',
                 'cnnmulti_1005_e2',
                 'cnnmulti_1005_e3',
                 'cnnmulti_1006_e1',
                 'cnnmulti_1006_e2',
                 'cnnmulti_1006_e3'
                 ]
        layers = 6

    if i == 3:
        files = ['mlpsingle_1004_e1',
                 'mlpsingle_1004_e2',
                 'mlpsingle_1004_e3',
                 'mlpsingle_1005_e1',
                 'mlpsingle_1005_e2',
                 'mlpsingle_1005_e3',
                 'mlpsingle_1006_e1',
                 'mlpsingle_1006_e2',
                 'mlpsingle_1006_e3'
                 ]
        layers = 2

    avg_GRS_shape = np.zeros(layers)
    avg_GRS_val = 0
    avg_GRS_valloss = 0
    avg_GRS_test = 0
    avg_GRS_testloss = 0
    avg_GRS_flop = 0
    avg_GRS_floploss = 0
    avg_GRS_para = 0
    avg_GRS_paraloss = 0

    avg_TQ_val = 0
    avg_TQ_valloss = 0
    avg_TQ_test = 0
    avg_TQ_testloss = 0
    avg_TQ_flop = 0
    avg_TQ_floploss = 0
    avg_TQ_para = 0
    avg_TQ_paraloss = 0

    avgModel = []
    model = None
    for fileName in files:
        model = fileName[0:9]
        ds = ''.join(['m',fileName[len(fileName)-5],fileName[len(fileName)-4],
                      fileName[len(fileName)-2],fileName[len(fileName)-1]])

        oriShape = None

        runs = 10
        title = None
        for i in range(runs):
            dataSet = []
            #print('comparisonEXPT/prunedM/'+fileName+'_seed0V'+str(i)+'_comparisonstats.csv')
            with open('outputs0602bw/regenPrunedM/'+fileName+'_V{'+str(i)+'}seed0_designEvaluation.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for j,row in enumerate(spamreader):
                    #print(row)
                    if j != 0:
                        oriShape = row[9]
                        #print(row[5])
                        grs_val, grs_valloss = accSapa(row[2])
                        grs_test, grs_testloss = accSapa(row[1])
                        grs_flop,grs_floploss = accSapa(row[3])
                        grs_para,grs_paraloss = accSapa(row[4])

                        tq_val, tq_valloss = accSapa(row[6])
                        tq_test, tq_testloss = accSapa(row[5])
                        tq_flop,tq_floploss = accSapa(row[7])
                        tq_para,tq_paraloss = accSapa(row[8])

                        dataSet.append(model)
                        dataSet.append(ds)
                        dataSet.append(i)
                        dataSet.append(grs_test)
                        dataSet.append(grs_testloss)
                        dataSet.append(grs_val)
                        dataSet.append(grs_valloss)
                        dataSet.append(grs_flop)
                        dataSet.append(grs_floploss)
                        dataSet.append(grs_para)
                        dataSet.append(grs_paraloss)

                        dataSet.append(tq_test)
                        dataSet.append(tq_testloss)
                        dataSet.append(tq_val)
                        dataSet.append(tq_valloss)
                        dataSet.append(tq_flop)
                        dataSet.append(tq_floploss)
                        dataSet.append(tq_para)
                        dataSet.append(tq_paraloss)

                        spamwriter.writerow(dataSet)

            csvfile.close()

outfile.close()
print('done')



