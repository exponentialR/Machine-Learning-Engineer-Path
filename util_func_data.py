def print_nge(numbers):
    """Find next greater element for ach of the n numbers
    Using nested for loop"""
    for i in range(0, len(numbers)):
        found = False
        for j in range(i+1, len(numbers)):
            if (numbers[j] > numbers[i]):
                found = True
                print(numbers[i], numbers[j])
                break
        if found == False: 
            print(numbers[i], -1)


def print_nge_stack(numbers):
    """Find next greater element for ach of the n numbers
    Using stack"""
    if len(numbers)==0:
        return
    
    stack = []
    stack.append(numbers[0])

    for i in range(1, len(numbers)):
        while len(stack)>0 and stack[-1] < numbers[i]:
            print(stack[-1], numbers[i])
            stack.pop
        stack.append(numbers[i])
    
    while len(stack)>0:
        print(stack[-1], -1)
        stack.pop()