'''
 author: xiaomin wu
 date: 1/16/2020
 '''
import csv


class ExtraCnnWeights:
    def __init__(self,folder):
        self.folder = folder

    def extractCnnLayerW(self,lw,name):
        # getting fileter weight
        with open("../"+self.folder+"/"+name+"Weight.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            wt = lw[0]
            for i in range(wt.shape[2]):
                for j in range(wt.shape[3]):
                    outstr = str(wt[0][0][i][j]) + ' ' \
                             + str(wt[0][1][i][j]) + ' ' \
                             + str(wt[1][0][i][j]) + ' ' \
                             + str(wt[1][1][i][j])
                    spamwriter.writerow([outstr])

        # getting bias each filter
        with open("../"+self.folder+"/"+name+"Bias.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            bias = lw[1]
            for i in range(bias.shape[0]):
                outstr = str(bias[i])
                spamwriter.writerow([outstr])

    def extractDenseLayerW(self,dw,name):
        # getting fileter weight
        with open("../"+self.folder+"/"+name+"Weight.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            wt = dw[0]
            for i in range(wt.shape[0]):
                outstr = ''
                for j in range(wt.shape[1]):
                    if j == (wt.shape[1] - 1):
                        outstr = outstr + str(wt[i][j])
                    else:
                        outstr = outstr +str(wt[i][j])+' '
                #outstr = outstr + ','
                spamwriter.writerow([outstr])

        # getting bias each filter
        with open("../"+self.folder+"/"+name+"Bias.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            bias = dw[1]
            outstr = ''
            for i in range(bias.shape[0]):
                if i == (bias.shape[0]-1):
                    outstr = outstr + str(bias[i])
                else:
                    outstr =outstr + str(bias[i]) + ' '
            spamwriter.writerow([outstr])

    def extractConv1Result(self,conv1,name):
        for f in range(conv1.shape[0]):
            with open("../"+self.folder+"/conv1out/"+name+"out"+str(f)+".csv", 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)

                for i in range(conv1.shape[3]):
                    outstr = ''
                    for j in range(conv1.shape[1]):
                        for k in range(conv1.shape[2]):
                            if j == (conv1.shape[1]-1) and k == (conv1.shape[2]-1):
                                outstr = outstr + str(conv1[f][j][k][i])
                            else:
                                outstr = outstr +str(conv1[f][j][k][i])+' '
                        #outstr = outstr + ','
                    if(outstr != ''):
                        spamwriter.writerow([outstr])

    def extractConv2Result(self,conv1,name):
        for f in range(conv1.shape[0]):
            with open("../"+self.folder+"/conv2out/"+name+"out"+str(f)+".csv", 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)

                for i in range(conv1.shape[3]):
                    outstr = ''
                    for j in range(conv1.shape[1]):
                        for k in range(conv1.shape[2]):
                            if j == (conv1.shape[1]-1) and k == (conv1.shape[2]-1):
                                outstr = outstr + str(conv1[f][j][k][i])
                            else:
                                outstr = outstr +str(conv1[f][j][k][i])+' '
                        #outstr = outstr + ','
                    if(outstr != ''):
                        spamwriter.writerow([outstr])

    def extractMaxPoolResult(self,conv1,name):
        for f in range(conv1.shape[0]):
            with open("../"+self.folder+"/maxPoolout/"+name+"out"+str(f)+".csv", 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)

                for i in range(conv1.shape[3]):
                    outstr = ''
                    for j in range(conv1.shape[1]):
                        for k in range(conv1.shape[2]):
                            if j == (conv1.shape[1] - 1) and k == (conv1.shape[2] - 1):
                                outstr = outstr + str(conv1[f][j][k][i])
                            else:
                                outstr = outstr +str(conv1[f][j][k][i])+' '
                        #outstr = outstr + ','
                    if(outstr != ''):
                        spamwriter.writerow([outstr])

    def extractdenseResult(self,dense,name):

        with open("../"+self.folder+"/outdense/"+name+"out.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)

            for i in range(dense.shape[0]):
                outstr = ''
                for j in range(dense.shape[1]):
                    if j == (dense.shape[1] -1):
                        outstr = outstr + str(dense[i][j])
                    else:
                        outstr = outstr + str(dense[i][j])+' '
                #outstr = outstr + ','
                spamwriter.writerow([outstr])

    def extractFlattenResult(self,flat,name):

        with open("../"+self.folder+"/"+name+"out.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)

            for i in range(flat.shape[0]):
                outstr = ''
                for j in range(flat.shape[1]):
                    if j == (flat.shape[1] - 1):
                        outstr = outstr + str(flat[i][j])
                    else:
                        outstr = outstr + str(flat[i][j])+' '
                #outstr = outstr + ','
                spamwriter.writerow([outstr])

    def extractTestXcnn(self,X_test):
        # extract testing data X
        with open("../"+self.folder+"/testInputXCNN.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for i in range(X_test.shape[0]):
                outstr = ''
                for j in range(X_test.shape[1]):
                    for k in range(X_test.shape[2]):
                        if j ==(X_test.shape[1]-1) and k ==(X_test.shape[2]-1):
                            outstr = outstr + str(X_test[i][j][k][0])
                        else:
                            outstr = outstr + str(X_test[i][j][k][0]) + ' '
                    #outstr = outstr + ','
                spamwriter.writerow([outstr])


    def extractTestXdense(self,X_test):
        # extract testing data X
        with open("../"+self.folder+"/testInputXDEN.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for i in range(X_test.shape[0]):
                outstr = ''
                for j in range(X_test.shape[1]):
                    if j == (X_test.shape[1] - 1):
                        outstr = outstr + str(X_test[i][j])
                    else:
                        outstr = outstr + str(X_test[i][j]) + ' '
                # outstr = outstr + ','
                spamwriter.writerow([outstr])

        with open("../" + self.folder + "/sampleDim.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            outstr = str(X_test.shape[1])
            spamwriter.writerow([outstr])


    def extractTestY(self,Y_test):
        with open("../"+self.folder+"/testInputY.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for i in range(Y_test.shape[0]):
                outstr = str(Y_test[i])
                spamwriter.writerow([outstr])

        with open("../" + self.folder + "/sampleSize.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            outstr = str(Y_test.shape[0])
            spamwriter.writerow([outstr])
