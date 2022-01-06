"""Given two sequences, find the length of the largest common subsequence. 
A sub-sequence should appear in both sequences in the same order, 
but they do not need to be continuous."""

def lcs(str1, str2):

    #get the length of both strings 

    p = len(str1)
    q = len(str2)

    #Initialise the 2D array (list) as 0 on every cell 

    LCS = [[0 for i in range(q+1)] for j in range(p+1)]

    #iterate over the 2d list 

    for i in range(0, p+1):
        for j in range(0, 1+q):
            if i ==0 or j ==0:
                LCS[i][j] = 0
            
            #if matches, increase the value by 1
            elif str1[i-1] == str2[j -1]:
                LCS[i][j] = 1 + LCS[i-1][j-1]
            
            else:
                LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1])
        return LCS[p][q]