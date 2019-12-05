# -*- coding: utf-8 -*-
"""

Read input files from testbed folder
Call function to write to dataset for each read input file

@author: jobose
"""

# encoding: utf-8
import os
import requests
from bs4 import BeautifulSoup
from collections import Counter
    
import extraction_ML_dataset_write

htmlfiles = [] #list to store all html files found at location
otherfiles = [] #list to keep any other file that do not match the criteria
    
def read_URLs_from_file(filename): 
    
    """Read URL from given file
    Pass files to the function that breaks into <p> tags and extracts features
    
    Parameters
    ----------
    filename : string
    Name of the file containing URLs.
    
    Returns
    -------
    None    
    """
    
    counter = 0

    file = open(filename, 'r')
    URLlines = file.readlines()
    
    for URLline in URLlines:
        print('Reading URL: ' + URLline)
        if(URLline.startswith('http') or URLline.startswith('https')):
            line = extraction_ML_dataset_write.read_URL(URLline)
            output_file_path = 'dataset_100URLs.csv'
            extraction_ML_dataset_write.write_to_output(output_file_path, 
                                                    line)
            counter = counter + 1
            
    file.close()
    
    print("Total files found:\t", counter)


def read_files_from_dir(location): 
    
    """Read input files from given location
    Pass files to the function that breaks into <p> tags and extracts features
    
    Parameters
    ----------
    location : string
    Location of the directory where the HTML files are stored on disk.
    
    Returns
    -------
    None    
    """
    
    counter = 0
    for file in os.listdir(location):
        try:
            if file.endswith(".html") or file.endswith('.htm'):
                print("html file found:\t", file)
                htmlfiles.append(str(file))
                line = extraction_ML_dataset_write.read_file(location + '/' + file)
                output_file_path = 'dataset.csv'
                extraction_ML_dataset_write.write_to_output(output_file_path, 
                                                            line)
                counter = counter + 1
    
            else:
                otherfiles.append(file)
                counter = counter + 1
                
        except Exception as e:
            raise e
            print("No files found here!")
    
    print("Total files found:\t", counter)

def main():
    path = '//chat-dev1/e/benchmark/Full_list_top1000/en-US_converted'
    read_files_from_dir(path)
    #read_URLs_from_file('100URLs.txt')
    
if __name__ == '__main__': 
    main()    