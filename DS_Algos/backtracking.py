#Function to check if the move is safe or not
def is_attack_safe(chess_board,row,col):
    #Checking if anything placed before this
    for item in range(col):
        if chess_board[row][item]==1:
            return False

    #Checking diagonals
    upper_diagonals = list(set(zip(range(row, -1, -1), range(col, -1, -1))))
    lower_diagonals = list(set(zip(range(row, len(chess_board), 1), range(col, -1, -1))))
    
    #Combining upper and lower diagonals
    diagonals = lower_diagonals+upper_diagonals
    for i,j in diagonals:
        if chess_board[i][j]==1:
            return False
    return True

#Chess board solver
def solver(chess_board,col):
    if col>=len(chess_board):
        return True
    for i in range(len(chess_board)):

        #If it safe to attack, place the move
        if is_attack_safe(chess_board,i,col):
            chess_board[i][col]=1
            if solver(chess_board,col+1):
                return True
            chess_board[i][col]=0
    return False

#Driver of N-Queen Solution
def NQueenSolution(chess_board):
    print('Board Size: ',len(chess_board))
    if solver(chess_board,0)==False:
        print('No Solution')
    else:
        for i in range(0,len(chess_board)):
                print(chess_board[i])
    print()