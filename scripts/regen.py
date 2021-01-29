import csv
import numpy as np
from FLOPs import ParaFlop

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
        modelType = 'cnn'

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
        modelType = 'mlp'

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
        modelType = 'cnn'

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
        modelType = 'mlp'

    for fileName in files:

        runs = 10
        for i in range(runs):
            #print('comparisonEXPT/prunedM/'+fileName+'_seed0V'+str(i)+'_comparisonstats.csv')
            with open('performEXPTbw/'+fileName+'_V{'+str(i)+'}seed0_designEvaluation.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                twoRows = []
                for j,row in enumerate(spamreader):
                    if j == 0:
                        twoRows.append(row)
                    else:
                        flopS,percentS = accSapa(row[3])
                        flopF,percentF = accSapa(row[7])
                        difference = flopS - flopF
                        #print(difference)
                        jsonfileGRS = 'outputs0602bw/modelinfo/'+fileName+'_V{'+str(i)+'}seed0model_GRS_pruned.json'
                        jsonfileOri = 'outputs0602bw/modelinfo/'+fileName+'_V{'+str(i)+'}seed0model_original.json'
                        trueFlopS,trueParaS = ParaFlop(jsonfileGRS,modelType)
                        trueFlopO,trueParaO = ParaFlop(jsonfileOri,modelType)
                        #print(trueFlopS)
                        #print(trueFlopO)
                        reducedS = float(trueFlopO - trueFlopS)*100/trueFlopO
                        redparaS = float(trueParaO - trueParaS)*100/trueParaO
                        #print(reducedS)
                        trueFlopF = trueFlopS - difference
                        reducedF = float(trueFlopO - trueFlopF)*100/trueFlopO
                        row[3] = accAssemInt(trueFlopS,reducedS)
                        row[4] = accAssemInt(trueParaS,redparaS)
                        row[7] = accAssemInt(trueFlopF,reducedF)
                        twoRows.append(row)
                        print(twoRows)

                        with open('outputs0602bw/regenPrunedM/'+fileName+'_V{'+str(i)+'}seed0_designEvaluation.csv', 'w', newline='') as outfile:
                            spamwriter = csv.writer(outfile, delimiter=',')
                            for k in range(len(twoRows)):
                                spamwriter.writerow(twoRows[k])
                        outfile.close()







