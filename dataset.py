# Python program to demonstrate  
# writing to file  
    
# Opening a file  
file1 = open('myfile.txt', 'w')  
    
# Writing a string to file  
file1.write(s)
  
def show_cwd_contents( path=".\dataset" ): 
    # A function that calls 2 functions to separately  
    # listing out directories and files. 
    # It takes a default argument as cwd(.). We can  
    # pass other paths too. 
    import os 
  
    f_list = [] 
  
    try: 
        for f in os.listdir(path): 
            if os.path.isfile(os.path.join(path, f)): 
                f_list.append(f+"\n") 
    except: 
        print ("\nError, once check the path")
        return
  
    return f_list
    
    
# Writing multiple strings  
# at a time  
file1.writelines(show_cwd_contents()) 

# Closing file  
file1.close()  
    
# Checking if the data is  
# written to file or not  
file1 = open('myfile.txt', 'r')  

print(file1.read())  
file1.close()  