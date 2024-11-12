#!/usr/bin/env python3


def main():
    file = open('maps/easygrid_base.txt', 'r')

    # put data into array
    lines = file.read().split("\n")
    chars = [list(line) for line in lines]
    
    print(chars)
    index = 0
    
    for l in range(0, len(chars)): 
        for c in range(0, len(chars[0])): # we're assuming that the map is rectangular
    
            if chars[l][c] == '-':
                place_agent(chars, l, c, index)
                index += 1
            else: 
                continue

def place_agent(chars, l, c, index):
    chars[l][c] = 'a'
    filename = 'maps2/easygrid_agent'+str(index)+'.txt'
    write_file(filename, chars)
    chars[l][c] = '-'
    
def write_file(filename, data):
    with open(filename, "w") as txt_file:
        for line in data:
            txt_file.write("".join(line) + "\n")
    
if __name__ == "__main__":
    main()