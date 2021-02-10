import time
import logging
import sys
import os

#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-9s) %(message)s',)

gpuNumber = 1
bbExe = "python bb.py"
singleOutputBB ='outputBB.txt'

logAllFile = "logAllOutputs.txt"


if __name__ == '__main__':

    input_file_name = sys.argv[1]  # The name of the file containing the coordinates, is passed as argument.
    
    syst_cmd = 'CUDA_VISIBLE_DEVICES=' +str(gpuNumber) + ' ' + bbExe + ' ' + input_file_name + ' > ' + singleOutputBB
    
    # print(syst_cmd)
    os.system(syst_cmd)
    
    with open(singleOutputBB, "r") as file:
        lines = file.readlines()
        lastLine = lines[len(lines)-1]
        output = lastLine.split("=")
        if output[0] == "Final_best_acc":
            bbOutput = output[1]
        file.close()
        # append file into log file
        append_cmd  =  'cat ' + input_file_name + ' >> ' + logAllFile + ' ; '
        append_cmd +=  'cat ' + singleOutputBB + ' >> ' + logAllFile
        os.system(syst_cmd)

        print(bbOutput, end=" ")
        

