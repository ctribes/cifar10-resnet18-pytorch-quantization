import threading
import time
import logging
import sys
import os

gpuList = [4,5,6,7]
bbExe = "python bb.py"
logAllFile = "logAllOutputs.txt"

class ThreadPool(object):
    def __init__(self):
        super(ThreadPool, self).__init__()
        self.active = []
        self.lock = threading.Lock()

    def makeActive(self, name, index):
        with self.lock:
            self.active.append(name)
            #logging.debug('Running: %s %s', self.active,str(index))

    def makeInactive(self, name, index):
        with self.lock:
            self.active.remove(name)
            #logging.debug('Ending: %s %s', self.active, str(index))


def startBBEval(s, pool, singleInputBB, singleOutputBB, lineContent):
    # logging.debug('Waiting to join the pool')
    with s:
        name = threading.currentThread().getName()
        if len(gpuList)==0:
            print("Houston we have a problem. The list of gpu is empty.")
        index = gpuList.pop(0)
        pool.makeActive(name,index)
        # time.sleep(2)
        #print(singleInputBB)
        #print(lineContent)
        fi = open(singleInputBB, "w")
        fi.write(lineContent)
        fi.close()

        syst_cmd = 'CUDA_VISIBLE_DEVICES=' +str(index) + ' ' + bbExe + ' ' + singleInputBB + ' > ' + singleOutputBB

        # print(syst_cmd)
        os.system(syst_cmd)

        pool.makeInactive(name,index)
        gpuList.insert(0,index)


if __name__ == '__main__':

    # Read the coordinates of x in a file passed as argument on the command line.
    input_file_name = sys.argv[1]  # The name of the file containing the coordinates,
    # is passed as argument.
    with open(input_file_name, 'r') as openfile:
        all = openfile.read()
        openfile.close()
        allLines = all.splitlines()

        singleOutputFileName = []
        singleInputFileName = []
        for i in range(len(allLines)):
            singleInputFileName.append("input."+str(i)+".txt")
            singleOutputFileName.append("output." + str(i) + ".txt")


        pool = ThreadPool()
        s = threading.Semaphore(len(gpuList))

        # Create all the threads in a list. Register their name in threadpool
        threads = [threading.Thread(target=startBBEval, name='thread_' + str(i), args=(s, pool, singleInputFileName[i],
                                                                             singleOutputFileName[i], allLines[i]))
                   for i in range(len(allLines))]

        # Start the threads
        for i in range(len(allLines)):
            #print("Start "+str(i)+ " "+str(gpuList))
            threads[i].start()

        # Wait for threads to complete before reading output
        bbOutput = ""
        for i in range(len(allLines)):
            threads[i].join()
            #print("Join "+str(i)+ " "+str(gpuList))

            with open(singleOutputFileName[i], "r") as file:
                lines = file.readlines()
                lastLine = lines[len(lines)-1]
                output = lastLine.split("=")
                if output[0] == "Final_best_acc":
                    bbOutput += output[1]
                file.close()
                # append file into log file
                append_cmd =  'cat ' + singleOutputFileName[i] + ' >> ' + logAllFile

                # print(syst_cmd)
                os.system(append_cmd)
                
        print(bbOutput)
        

