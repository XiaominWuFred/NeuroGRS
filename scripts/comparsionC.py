import csv
import numpy as np

ccsv = []


for i in range(2):
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

    for fileName in files:
        model = fileName[0:9]
        ds = ''.join(['m', fileName[len(fileName) - 5], fileName[len(fileName) - 4],
                      fileName[len(fileName) - 2], fileName[len(fileName) - 1]])
        GRS_shape = np.zeros(layers)
        GRS_val = 0
        RRS_shape = np.zeros(layers)
        RRS_val = 0
        NWM_shape = np.zeros(layers)
        NWM_val = 0

        oriShape = None
        dataSet = []

        for i in range(10):
            #print('comparisonEXPT/prunedM/'+fileName+'_seed0V'+str(i)+'_comparisonstats.csv')
            with open('comparisonEXPT/prunedM/'+fileName+'_seed0V'+str(i)+'_comparisonstats.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for i,row in enumerate(spamreader):
                    #print(row)
                    if i != 0:
                        oriShape = row[1]
                        GRS_val = GRS_val + float(row[5])
                        GRS_shape[0] += int(row[2].split('X')[0])
                        GRS_shape[1] += int(row[2].split('X')[1])
                        GRS_shape[2] += int(row[2].split('X')[2])
                        GRS_shape[3] += int(row[2].split('X')[3])

                        #print(int(row[2].split('X')[0]))

                        RRS_val = RRS_val + float(row[6])
                        RRS_shape[0] += int(row[3].split('X')[0])
                        RRS_shape[1] += int(row[3].split('X')[1])
                        RRS_shape[2] += int(row[3].split('X')[2])
                        RRS_shape[3] += int(row[3].split('X')[3])


                        NWM_val = NWM_val + float(row[7])
                        NWM_shape[0] += int(row[4].split('X')[0])
                        NWM_shape[1] += int(row[4].split('X')[1])
                        NWM_shape[2] += int(row[4].split('X')[2])
                        NWM_shape[3] += int(row[4].split('X')[3])

            csvfile.close()


        print(fileName)
        GRS_val = GRS_val/10
        print('GRSAvgVal: '+str(GRS_val))

        RRS_val = RRS_val/10
        print('RRSAvgVal: '+str(RRS_val))

        NWM_val = NWM_val/10
        print('NWMAvgVal: '+str(NWM_val))

        GRS_shape = GRS_shape/10
        RRS_shape = RRS_shape/10
        NWM_shape = NWM_shape/10

        tmp = []
        for i in range(len(GRS_shape)-1):
            tmp.append(str(GRS_shape[i]))
        grsStr = 'X'.join(tmp)
        print('GRSshape: '+grsStr)

        tmp = []
        for i in range(len(RRS_shape)-1):
            tmp.append(str(RRS_shape[i]))
        rrsStr = 'X'.join(tmp)
        print('RRSshape: '+rrsStr)

        tmp = []
        for i in range(len(NWM_shape)-1):
            tmp.append(str(NWM_shape[i]))
        nwmStr = 'X'.join(tmp)
        print('NWMshape: '+nwmStr)
        print('')
        dataSet.append(model)
        dataSet.append(ds)
        dataSet.append(oriShape)
        dataSet.append(grsStr)
        dataSet.append(rrsStr)
        dataSet.append(nwmStr)
        dataSet.append(GRS_val)
        dataSet.append(RRS_val)
        dataSet.append(NWM_val)
        ccsv.append(dataSet)

print('done')
with open('comparisonC.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['model','dataset','shape_S','shpae_P_G','shape_P_R','shape_P_N','grsVal','rrsVal','nwmVal'])
    for i in range(len(ccsv)):
        spamwriter.writerow(ccsv[i])


