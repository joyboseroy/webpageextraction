# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:20:12 2018

Read input file
Extract features
Write extracted features to dataset 

@author: jobose
"""

#import extraction_ML_dataset_read

import requests
import re

# encoding: utf-8
from bs4 import BeautifulSoup, SoupStrainer
from collections import Counter
#from HMTLParser import HTMLParser
    
def read_file(HTML_filename): 
    """Read file from input directory and call parsing function
    
    Parameters
    ----------
    HTML_filename : string
    Filename of the input HTML file to parse.
    
    Returns
    -------
    line: string
    Returns string containing the extracted features from input HTML file
    """
    line = parse_file(HTML_filename)
    if line != None: 
        return HTML_filename + ',' + line
    else:
        return ''

def read_URL(URLname):
    """Read URL line from file and call parsing function
    
    Parameters
    ----------
    URLname : string
    URL of the input HTML file to parse.
    
    Returns
    -------
    line: string
    Returns string containing the extracted features from input HTML file
    """
    line = parse_URL(URLname)
    if line != None: 
        return URLname + ',' + line
    else:
        return ''

    
def parse_URL(URLname):
    """Use BeautifulSoup to parse the URL
    Call function to extract features
    
    Parameters
    ----------
    URLname : string
    URL of the HTML file to parse.
    
    Returns
    -------
    line: string
    Returns string containing the extracted features from input HTML file
    """
    #If page is URL then use page.content for BeautifulSoup
    page = requests.get(URLname, stream = True)
    soup = BeautifulSoup(page.content, 'lxml') #, parse_only = strainer) #'html.parser')
    #can also call BeautifulSoup(html)
    #mylist = soup.select('div p')
    line = extract_features(soup)
    return line

    
def parse_file(HTML_file):
    """Use BeautifulSoup to parse the file
    Call function to extract features
    
    Parameters
    ----------
    HTML_file : string
    Path and filename of the input HTML file to parse.
    
    Returns
    -------
    line: string
    Returns string containing the extracted features from input HTML file
    """
    
    #If page is filename then use open
    soup = BeautifulSoup(open(HTML_file, encoding = 'utf-8'), 'html.parser')
    #can also call BeautifulSoup(html)
    #mylist = soup.select('div p')
    line = extract_features(soup)
    return line
    
"""
Mozilla readability

 REGEXPS: {

    unlikelyCandidates: /-ad-|banner|breadcrumbs|combx|comment|community|cover-wrap|disqus|extra|foot|header|legends|menu|related|remark|replies|rss|shoutbox|sidebar|skyscraper|social|sponsor|supplemental|ad-break|agegate|pagination|pager|popup|yom-remote/i,

    okMaybeItsACandidate: /and|article|body|column|main|shadow/i,

    positive: /article|body|content|entry|hentry|h-entry|main|page|pagination|post|text|blog|story/i,

    negative: /hidden|^hid$| hid$| hid |^hid |banner|combx|comment|com-|contact|foot|footer|footnote|masthead|media|meta|outbrain|promo|related|scroll|share|shoutbox|sidebar|skyscraper|sponsor|shopping|tags|tool|widget/i,

    extraneous: /print|archive|comment|discuss|e[\-]?mail|share|reply|all|login|sign|single|utility/i,

    byline: /byline|author|dateline|writtenby|p-author/i,

    replaceFonts: /<(\/?)font[^>]*>/gi,

    normalize: /\s{2,}/g,

    videos: /\/\/(www\.)?((dailymotion|youtube|youtube-nocookie|player\.vimeo|v\.qq)\.com|(archive|upload\.wikimedia)\.org|player\.twitch\.tv)/i,

    nextLink: /(next|weiter|continue|>([^\|]|$)|»([^\|]|$))/i,

    prevLink: /(prev|earl|old|new|<|«)/i,

    whitespace: /^\s*$/,

    hasContent: /\S$/,

  },
"""    

def extract_features(soup):
    """Extract each <p> tag separately
    Get features: word count, link count, word density, link density
    Write the extracted features to output dataset file
    TODO: Get features used in chrome and mozilla
    
    Parameters
    ----------
    soup : BeautifulSoup object 
    Contains parsed components e.g. div elements, p tags, a tags etc
    extracted from the HTML file
    
    Returns
    -------
    line: string
    Returns string containing the extracted features from input HTML file
    """
    #mylistnew = soup.find_all('-ad-|banner|breadcrumbs|combx|comment|community|cover-wrap|disqus|extra|foot|header|legends|menu|related|remark|replies|rss|shoutbox|sidebar|skyscraper|social|sponsor|supplemental|ad-break|agegate|pagination|pager|popup|yom-remote/i,

    
    mylist = soup.find_all(re.compile(r'h1|h2|h3|h4|p|img')) #('p')find all div elements
    item_count = 0
    for items in mylist:
        item_count = item_count + 1
        link_count = 0
        for link in items.find_all('a'):
            link_count = link_count + 1
        
        word_count = len(items.text.split())
        input_count = len(items.find_all('input'))
        script_count = len(items.find_all('script'))
        
        link_input_script_count = link_count + input_count + script_count
        links_plus_words = link_input_script_count + word_count
        if links_plus_words != 0:
            link_density = link_input_script_count/links_plus_words
            text_density = word_count/links_plus_words
        else:
            link_density = 0
            text_density = 0
        
        print('Number of words =' , word_count)
        print('number of links and inputs and scripts=', 
              link_input_script_count) #len(items.find_all('a')))
        if links_plus_words != 0:
            print('link density = ', 
              link_input_script_count / links_plus_words)
            print('text density = ', 
              word_count / links_plus_words)
        #print('ads = ', len(items.find_all('adsbygoogle')))
        print(items)
        print('\n')
        
        line = str(item_count) + ',' + \
               str(word_count) + ',' + \
               str(input_count) + ',' + \
               str(script_count) + ',' + \
               str(link_input_script_count) + ',' + \
               str(round(text_density, 2)) + ',' + \
               str(round(link_density, 2)) + '\n'
        #print(line)
        return(line)

def write_to_output(output_file_path, line):
    """Write to the dataset file
    
    Parameters
    ----------
    output_file_path : string
    Path and filename of the output file.
    
    line : string
    Line to write to the output file.

    Returns
    -------
    None    
    """
    
    if line != None:
        with open(output_file_path, 'a', encoding = 'utf-8') as outfile:
            outfile.write(line)
    
def main():
    #pageURL = 'http://locusmag.com/2016/03/cory-doctorow-wealth-inequality-is-even-worse-in-reputation-economies/'
    #line = read_URL(pageURL)
    
    HTML_file = 'myfileread.html'
    line = read_file(HTML_file)
    
    output_file_path = 'dataset_100URLs.csv'
    write_to_output(output_file_path, line)
        
if __name__ == '__main__': 
    main()    