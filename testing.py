
#!/usr/bin/python

import os
import subprocess


def main():
    
    for kernelSize in range(3,32,2):  
        logFile = "log" + str(kernelSize) + ".txt"
        cudacmd = ["nvprof", "--metrics", "gld_transactions", "--log-file" ,logFile, "./conv", str(kernelSize),  "image.pgm"]
        cuda = subprocess.Popen(cudacmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cudaout=cuda.communicate()
        for line in cudaout:
            if(line.rfind("Passed")!=-1):
                print "Kernel Size ", kernelSize, " Passed!!"
              

if __name__ == "__main__":main()
