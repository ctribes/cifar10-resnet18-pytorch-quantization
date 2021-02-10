import time
import logging
import sys
import os

#logging.basicConfig(level=logging.DEBUG,
#                    format='(%(threadName)-9s) %(message)s',)

gpuList = [0,1,2,3]
bbExe = "python bb.py"
logAllFile = "logAllOutputs.txt"


if __name__ == '__main__':

    # Get the input file name passed as an argument
    input_file_name = sys.argv[1]  # The name of the file containing the coordinates,
    # is passed as argument.
    
    # Nomad puts the parallel evaluation index in the name of the input.
    # Get this index
    parts = input_file_name.split(".")
    index = int(parts[-1])
    
    # The evaluation index is also used to create a unique output file for each evaluation
    singleOutputBB ='outputBB.'+parts[-1]+'.txt'
    
    # This evaluation is given a GPU device according to its index
    syst_cmd = 'CUDA_VISIBLE_DEVICES=' +str(gpuList[index]) + ' ' + bbExe + ' ' + input_file_name + ' > ' + singleOutputBB
    
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

        # Print the output, Nomad redirects this into a file and reads the results
        print(bbOutput, end=" ")

