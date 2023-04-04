# Program to generate a distribution

# imports of external packages to use in our code
import sys
import numpy as np

# import our Random class from python/Random.py file
sys.path.append(".")

# main function for our coin toss Python code
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [-seed number], please manually change the probabilities in the Cate definition if needed" % sys.argv[0])
        print
        sys.exit(1)

    # default seed
    seed = 8888

    # default single coin-toss probability for each side
    rare = 0.02
    prob2 = 0.1
    prob3 = 0.6
    p1 = 0.25
    p2 = 0.001
    #number of rolls needed before the super rare probability increases
    pity = 50
    # number of coin tosses (per experiment)
    Ntoss = 5000

    # number of experiments
    Nexp = 1

    # output file defaults
    doOutputFile = True
    # call a few counters
    b = 0
    limited = 0
    abnormal = 0
    prob1 = rare
    OutputFileName = 'H0.txt'
    # class instance of our Random class using seed
    random = Random(seed)
    if doOutputFile: #Write out the data into a file
        outfile = open(OutputFileName, 'w')
        for e in range(0,Nexp):
            for t in range(0,Ntoss):
                N = random.Cate(prob1,prob2,prob3) #Select a random value according to the categorical distribution
                outfile.write(str(N)+" ")
                if N == 0:
                  M = random.Bern(p1) #If the value falls into the "super rare" rate, compute whenever the roll is a limited-time one or not
                  outfile.write(str(M)+" ")
                  if M == 5: #Count the limited-time that you get
                    limited = limited +1
                  b = 0 #Set the pity timer back to 0
                  prob1 = rare
                if N == 3:
                  M = random.Bern(p2) +2 #See if for every "common" rolls, calculate the probably to get the abnormal character
                  outfile.write(str(M)+" ")
                  if M == 7:
                    abnormal = abnormal + 1 #count the number of abnormal characters you get
                  b = b+1 #still count up the pity for the super-rare roll
                else:
                  b = b+1 #If the value is not a "super rare" rate, count the pity up one
                if b > pity:
                  prob1 = prob1+0.02 #If the pity is past the specified number, start adding 2% to the "rate up" probability
            outfile.write(str(limited)+" ")
            limited = 0
            outfile.write(" \n")
            b = 0
        outfile.close()
    else:
        for e in range(0,Nexp):
            for t in range(0,Ntoss):
                N = random.Cate(prob1,prob2,prob3) #Select a random value according to the categorical distribution
                print(N, end=' ')
                if N == 0:
                  M = random.Bern(p1) #If the value falls into the "super rare" rate, compute whenever the roll is a limited-time one or not
                  print("(", M, ")", end=' ')
                  if M == 5: #Count the limited-time that you get
                    limited = limited +1
                  b = 0 #Set the pity timer back to 0
                  prob1 = rare
                if N == 3:
                  M = random.Bern(p2) +2 #See if for every "common" rolls, calculate the probably to get the abnormal character
                  print("(", M, ")", end=' ')
                  if M == 7:
                    abnormal = abnormal + 1 #count the number of abnormal characters you get
                  b = b+1 #still count up the pity for the super-rare roll
                else:
                  b = b+1 #If the value is not a "super rare" rate, count the pity up one
                if b > pity:
                  prob1 = prob1+0.02 #If the pity is past the specified number, start adding 2% to the "rate up" probability
            print("Number of limited-time rolled is :"+str(limited)+"  ")
            limited = 0
            b = 0
            print(" ")
