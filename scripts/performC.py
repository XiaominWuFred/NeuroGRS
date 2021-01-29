import csv
import numpy as np

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
        GRS_shape = np.zeros(layers)
        GRS_val = 0
        GRS_valloss = 0
        GRS_test = 0
        GRS_testloss = 0
        GRS_flop = 0
        GRS_floploss = 0
        GRS_para = 0
        GRS_paraloss = 0

        TQ_val = 0
        TQ_valloss = 0
        TQ_test=0
        TQ_testloss =0
        TQ_flop=0
        TQ_floploss =0
        TQ_para=0
        TQ_paraloss=0



        oriShape = None
        dataSet = []
        runs = 10
        title = None
        for i in range(runs):
            #print('comparisonEXPT/prunedM/'+fileName+'_seed0V'+str(i)+'_comparisonstats.csv')
            with open('outputs0602bw/regenPrunedM/'+fileName+'_V{'+str(i)+'}seed0_designEvaluation.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i,row in enumerate(spamreader):
                    #print(row)
                    if i != 0:
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

                        GRS_val += grs_val
                        GRS_valloss +=  grs_valloss
                        GRS_test += grs_test
                        GRS_testloss += grs_testloss
                        GRS_flop += grs_flop
                        GRS_floploss += grs_floploss
                        GRS_para += grs_para
                        GRS_paraloss += grs_paraloss

                        TQ_val += tq_val
                        TQ_valloss +=  tq_valloss
                        TQ_test += tq_test
                        TQ_testloss += tq_testloss
                        TQ_flop += tq_flop
                        TQ_floploss += tq_floploss
                        TQ_para += tq_para
                        TQ_paraloss += tq_paraloss

                        for i in range(layers):
                            GRS_shape[i] += int(row[10].split('X')[i])

            csvfile.close()


        print(fileName)
        GRS_val = GRS_val/runs
        GRS_valloss = GRS_valloss/runs
        GRS_test=GRS_test/runs
        GRS_testloss=GRS_testloss/runs
        GRS_flop=GRS_flop/runs
        GRS_floploss=GRS_floploss/runs
        GRS_para=GRS_para/runs
        GRS_paraloss=GRS_paraloss/runs

        TQ_val = TQ_val/runs
        TQ_valloss = TQ_valloss/runs
        TQ_test=TQ_test/runs
        TQ_testloss=TQ_testloss/runs
        TQ_flop=TQ_flop/runs
        TQ_floploss=TQ_floploss/runs
        TQ_para=TQ_para/runs
        TQ_paraloss=TQ_paraloss/runs

        GRS_shape = GRS_shape / runs

        #cumulative for averaging model on 9 datasets
        avg_GRS_val += GRS_val
        avg_GRS_valloss += GRS_valloss
        avg_GRS_test += GRS_test
        avg_GRS_testloss += GRS_testloss
        avg_GRS_flop += GRS_flop
        avg_GRS_floploss += GRS_floploss
        avg_GRS_para += GRS_para
        avg_GRS_paraloss += GRS_paraloss

        avg_TQ_val += TQ_val
        avg_TQ_valloss += TQ_valloss
        avg_TQ_test += TQ_test
        avg_TQ_testloss += TQ_testloss
        avg_TQ_flop += TQ_flop
        avg_TQ_floploss += TQ_floploss
        avg_TQ_para += TQ_para
        avg_TQ_paraloss += TQ_paraloss

        avg_GRS_shape += GRS_shape
        ##############################################

        testacc_s = accAssem(GRS_test,GRS_testloss)
        valacc_s = accAssem(GRS_val,GRS_valloss)
        flops_s = accAssemInt(GRS_flop,100-GRS_floploss)
        paras_s = accAssemInt(GRS_para,100-GRS_paraloss)

        testacc_f = accAssem(TQ_test,TQ_testloss)
        valacc_f = accAssem(TQ_val,TQ_valloss)
        flops_f = accAssemInt(TQ_flop,100-TQ_floploss)
        paras_f = accAssemInt(TQ_para,100-TQ_paraloss)

        tmp = []
        for i in range(len(GRS_shape)-1):
            tmp.append(str(GRS_shape[i]))
        grsStr = 'X'.join(tmp)
        print('GRSshape: '+grsStr)

        print('')
        dataSet.append(model)
        dataSet.append(ds)
        dataSet.append(testacc_s)
        dataSet.append(valacc_s)
        dataSet.append(flops_s)
        dataSet.append(paras_s)
        dataSet.append(grsStr)
        dataSet.append(testacc_f)
        dataSet.append(valacc_f)
        dataSet.append(flops_f)
        dataSet.append(paras_f)


        ccsv.append(dataSet)

    #finished 9 files for one model
    numOfFiles = len(files)
    print("numOfFiles: "+str(numOfFiles))
    avg_GRS_val = avg_GRS_val / numOfFiles
    avg_GRS_valloss = avg_GRS_valloss / numOfFiles
    avg_GRS_test = avg_GRS_test / numOfFiles
    avg_GRS_testloss = avg_GRS_testloss / numOfFiles
    avg_GRS_flop = avg_GRS_flop / numOfFiles
    avg_GRS_floploss = avg_GRS_floploss / numOfFiles
    avg_GRS_para = avg_GRS_para / numOfFiles
    avg_GRS_paraloss = avg_GRS_paraloss / numOfFiles

    avg_TQ_val = avg_TQ_val / numOfFiles
    avg_TQ_valloss = avg_TQ_valloss / numOfFiles
    avg_TQ_test = avg_TQ_test / numOfFiles
    avg_TQ_testloss = avg_TQ_testloss / numOfFiles
    avg_TQ_flop = avg_TQ_flop / numOfFiles
    avg_TQ_floploss = avg_TQ_floploss / numOfFiles
    avg_TQ_para = avg_TQ_para / numOfFiles
    avg_TQ_paraloss = avg_TQ_paraloss / numOfFiles

    avg_GRS_shape = avg_GRS_shape / numOfFiles

    avg_testacc_s = accAssem(avg_GRS_test, avg_GRS_testloss)
    avg_valacc_s = accAssem(avg_GRS_val, avg_GRS_valloss)
    avg_flops_s = accAssemInt(avg_GRS_flop, 100 - avg_GRS_floploss)
    avg_paras_s = accAssemInt(avg_GRS_para, 100 - avg_GRS_paraloss)

    avg_testacc_f = accAssem(avg_TQ_test, avg_TQ_testloss)
    avg_valacc_f = accAssem(avg_TQ_val, avg_TQ_valloss)
    avg_flops_f = accAssemInt(avg_TQ_flop, 100 - avg_TQ_floploss)
    avg_paras_f = accAssemInt(avg_TQ_para, 100 - avg_TQ_paraloss)

    tmp = []
    for i,each in enumerate(avg_GRS_shape):
        avg_GRS_shape[i] = round(each,1)

    for i in range(len(avg_GRS_shape) - 1):
        tmp.append(str(avg_GRS_shape[i]))
    avg_grsStr = 'X'.join(tmp)
    print('GRSshape: ' + avg_grsStr)

    avgModel.append(model)
    avgModel.append(avg_testacc_s)
    avgModel.append(avg_valacc_s)
    avgModel.append(avg_flops_s)
    avgModel.append(avg_paras_s)
    avgModel.append(avg_grsStr)
    avgModel.append(avg_testacc_f)
    avgModel.append(avg_valacc_f)
    avgModel.append(avg_flops_f)
    avgModel.append(avg_paras_f)

    avg_ccsv.append(avgModel)


print('done')

with open('performCbwRegen.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['model','Dataset','TestAcc_S(lost%)','ValAcc_S(lost%)','FLOPs_S(% of initial)','Paras_S(% of initial)','Structure','TestAcc_F(lost%)','ValAcc_F(lost%)','FLOPs_F(% of initial)','Paras_F(% of initial)'])
    for i in range(len(ccsv)):
        spamwriter.writerow(ccsv[i])
csvfile.close()

with open('performAvgModelRegen.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['model','TestAcc_S(lost%)','ValAcc_S(lost%)','FLOPs_S(% of initial)','Paras_S(% of initial)','Structure','TestAcc_F(lost%)','ValAcc_F(lost%)','FLOPs_F(% of initial)','Paras_F(% of initial)'])
    for i in range(len(avg_ccsv)):
        spamwriter.writerow(avg_ccsv[i])
csvfile.close()


