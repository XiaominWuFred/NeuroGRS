import os
import csv

CPUONLY = True

os.system("rm ./originalOnly.csv")
outfile = open('originalOnly.csv', 'a', newline='')
spamwriter = csv.writer(outfile, delimiter=',')
if CPUONLY:
    spamwriter.writerow(['Model', 'Dataset', 'CPU_O (ms)'])
else:
    spamwriter.writerow(['Model','Dataset','CPU_O (ms)', 'GPU_O (ms)'])
outfile.close()

for i in range(4):

    if i == 0:
        files = ['cnnsingle_1004_e1',
                 'cnnsingle_1005_e1',
                 'cnnsingle_1006_e1'
                 ]
        layers = 4
        modelType = 'cnn'
    if i ==1:
        files = ['mlpmulti_1004_e1',
                 'mlpmulti_1005_e1',
                 'mlpmulti_1006_e1'
                 ]
        layers = 4
        modelType = 'mlp'

    if i ==2:
        files = ['cnnmulti_1004_e1',
                 'cnnmulti_1005_e1',
                 'cnnmulti_1006_e1'
                 ]
        layers = 6
        modelType = 'cnn'

    if i == 3:
        files = ['mlpsingle_1004_e1',
                 'mlpsingle_1005_e1',
                 'mlpsingle_1006_e1'
                 ]
        layers = 2
        modelType = 'mlp'


    for fileName in files:
        model = fileName.split('_')[0]
        ds = ''.join(['m',fileName[len(fileName)-5],fileName[len(fileName)-4],
                      fileName[len(fileName)-2],fileName[len(fileName)-1]])
        dsNoM = ''.join([fileName[len(fileName)-5],fileName[len(fileName)-4],
                      fileName[len(fileName)-2],fileName[len(fileName)-1]])

        if CPUONLY:
            runtime_cpu_ori_10 = 0

        else:
            runtime_cpu_ori_10 = 0
            runtime_gpu_ori_10 = 0


        record = []
        runs = 1
        for i in range(runs):

            model = model #sys.argv[1]
            fileNameSet = ['',fileName.split('_')[1],fileName.split('_')[2],'V{'+str(i)+'}seed0']
            fileNameToConvert = '_'.join(fileNameSet) #sys.argv[2]
            set = dsNoM #sys.argv[3]
            path = '../../../NeuroGRSoutputSave/outputs0602bw/regenModelinfo/'  # sys.argv[4]

            #original model
            Pruned = '0' # sys.argv[5]  # 'False' original, 'True' pruned
            os.system('cd ../graphGen/src/\npython3 neuroGRS_convert.py '+model+' '+fileNameToConvert+' '+set+' '+path+' '+Pruned)
            os.system('cd ../dnn_inference_CUDA_C/lide_c_dnnlayers/test_c/autoGenModel\nbash make.bash\nbash run100.bash > times100')
            f = open('../dnn_inference_CUDA_C/lide_c_dnnlayers/test_c/autoGenModel/times100','r')
            runtime_cpu_ori = float(f.readline().split()[3])
            runtime_cpu_ori_10 += runtime_cpu_ori
            f.close()
            if CPUONLY:
                pass
            else:
                os.system('cd ../dnn_inference_CUDA_C/lide_c_dnnlayers/test_cuda/autoGenModel\nbash make.bash\nbash run100.bash > times100')
                f = open('../dnn_inference_CUDA_C/lide_c_dnnlayers/test_cuda/autoGenModel/times100', 'r')
                runtime_gpu_ori = float(f.readline().split()[3])
                runtime_gpu_ori_10 += runtime_gpu_ori
                f.close()

        if CPUONLY:
            runtime_cpu_ori_10 = runtime_cpu_ori_10/runs
            record.append(model)
            record.append(ds)
            record.append(runtime_cpu_ori_10)
        else:
            runtime_cpu_ori_10 = runtime_cpu_ori_10/runs
            runtime_gpu_ori_10 = runtime_gpu_ori_10/runs
            record.append(model)
            record.append(ds)
            record.append(runtime_cpu_ori_10)
            record.append(runtime_gpu_ori_10)


        outfile = open('originalOnly.csv', 'a', newline='')
        spamwriter = csv.writer(outfile, delimiter=',')
        spamwriter.writerow(record)
        outfile.close()

print('done')




