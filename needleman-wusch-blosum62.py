import numpy as np
import pandas as pd

def blosum62():
    index = np.array(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S',
    'T','W','Y','V'])
    
    score = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    [1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    [0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]
    ])
    
    blosum62 = pd.DataFrame(score,index=index,columns=index)
    return blosum62  

def matrix(input1 , input2):
    row                 = len(input2)+1    
    column              = len(input1)+1  
    init_matrix         = np.ndarray(shape =(row,column) , dtype=int )
    traceback_matrix    = np.ndarray(shape =(row,column) , dtype=object )
    for i in range(row):
        for j in range(column):
            init_matrix[0][0] = 0
            traceback_matrix[0][0] = "done"
            if(i==0 and j!=0):
                init_matrix[i][j]       = init_matrix[i][j-1]-6
                traceback_matrix[i][j]  = "left"
            elif(j==0 and i!=0):
                init_matrix[i][j]       = init_matrix[i-1][j]-6
                traceback_matrix[i][j]  = "up"
            else:
                up                  = init_matrix[i-1][j]-6                
                left                = init_matrix[i][j-1]-6
                diagonal            = init_matrix[i-1][j-1] + blosum62_matrix[input2[i-1]][input1[j-1]] 
                concatenate         = [up,left,diagonal]                
                init_matrix[i][j] = max(concatenate)
                if(concatenate.index(init_matrix[i][j])== 0):
                    traceback_matrix[i][j] = "up"
                elif(concatenate.index(init_matrix[i][j])== 1):
                    traceback_matrix[i][j] = "left"
                else:
                    traceback_matrix[i][j] = "diagonal"
    matrix_index        = [""] + [idx for idx in input2]
    matrix_column       = [""] + [clm for clm in input1]
            
    return init_matrix,traceback_matrix,matrix_index,matrix_column
    
def best_aligment(s1,s2,input):
    i       = input.shape[0]-1
    j       = input.shape[1]-1
    aligment1 = []
    aligment2 = []
    while(i>0 or j>0):
        if(input[i][j] == "diagonal"):
            i -= 1
            j -= 1
            aligment1.append(s1[j])
            aligment2.append(s2[i])
        elif(input[i][j] == "left"):
            j -= 1
            aligment1.append(s1[j])
            aligment2.append("-")
        else:
            i -= 1
            aligment1.append("-")
            aligment2.append(s2[i])
    
    aligment1.reverse(),aligment2.reverse()
    return aligment1,aligment2

     
blosum62_matrix     = blosum62()
sequence1           = "MNALQM"
sequence2           = "NALMSQA"


score_matrix        = pd.DataFrame(matrix(sequence1,sequence2)[0],
                                   index = matrix(sequence1,sequence2)[2] , 
                                    columns = matrix(sequence1,sequence2)[3])
                                   
traceback_matrix    = pd.DataFrame(matrix(sequence1,sequence2)[1],
                                   index = matrix(sequence1,sequence2)[2], 
                                   columns = matrix(sequence1,sequence2)[3])
                                   
result              = best_aligment(sequence1,sequence2,matrix(sequence1,sequence2)[1])

print blosum62_matrix

print score_matrix

print traceback_matrix

print result