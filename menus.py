# -*- coding: utf-8 -*-
"""
Nutrislice Extractor
Created on Mon Oct 28 08:32:54 2019
@author: carverjc

This software is used to extract menu items from Nutrislice Menus. See README.md for details on usage.
"""

import cv2, sys
import pytesseract
import numpy as np
import os
import pandas as pd
import skimage
import time
from skimage import io
import copy
from matplotlib import pyplot as plt
from os.path import join
from os import makedirs
from glob import glob

# MUST CHANGE THESE TWO
os.environ["TESSDATA_PREFIX"] = "PATH_TO_TESSDATA"
pytesseract.pytesseract.tesseract_cmd = "PATH_TO_tesseract.exe"

def find_lines (img, length_of_run = 20, distance = 100):
    runs = [(-1)*(distance + 1)]
    for i in range(IMG_WIDTH):
        for j in range(IMG_HEIGHT):
            run_length = 0
            if np.all(img[j,i] == 0.0) and i - runs[-1] > distance:
                for run in range(length_of_run):
                    try:
                        if np.all(img[j + run, i] == 0.0):
                            run_length += 1
                    except IndexError:
                        break
                if run_length == length_of_run:
                    runs.append(i)
                    break
    return runs[1:] #list(dict.fromkeys(runs))

def greatest_line (img):
    IMG_WIDTH = img.shape[:2][1]
    IMG_HEIGHT = img.shape[:2][0]
    max_list = []
    for i in range(IMG_WIDTH):
        total_col_max = 0
        for j in range(IMG_HEIGHT):
            max_run = 0
            if np.all(img[j,i] == 0.0):
                new_index = j
                try:
                    while np.all(img[new_index,i] == 0.0):
                        max_run += 1
                        new_index += 1
                except IndexError:
                    continue
                if max_run > total_col_max:
                    total_col_max = max_run
        max_list.append(total_col_max)
    return max_list

def calculate_pixels (img, find_row = True, derivative = False):
    row_mean = []
    if find_row == True:
        for i in range(IMG_HEIGHT):
            intermediate_sum = 0
            for j in range(IMG_WIDTH):
                intermediate_sum = intermediate_sum + img[i,j][0]
            row_mean.append(intermediate_sum / IMG_WIDTH)
    else:
        for i in range(IMG_WIDTH):
            intermediate_sum = 0
            for j in range(IMG_HEIGHT):
                intermediate_sum = intermediate_sum + img[j,i][0]
            row_mean.append(intermediate_sum / IMG_HEIGHT)
    
    if derivative == True:
        for i in range(len(row_mean) - 1):
            row_mean[i] = row_mean[i + 1] - row_mean[i]
        row_mean = row_mean[:-1]
    return row_mean

def plot_df (df, title="", xlabel='Pixel Index', ylabel='Pixel Value', dpi=100):
    df = pd.DataFrame(df)
    df.index.name = xlabel
    df.reset_index(inplace=True)
    df.columns = [xlabel, ylabel]
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(df[xlabel], df[ylabel], color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.title("Mean Horizontal Pixel Value From Top of Image")
    plt.show()

def cut_finder (df, max_pixel, distance, find_black = True):
    cuts = []
    cuts = [(-1)*distance]
    for i in range(len(df)):
        if find_black:
            if df[i] < max_pixel and (i - cuts[-1]) > distance:
                cuts.append(i)
        else:
            if df[i] < max_pixel and (i - cuts[-1]) > distance:
                if len(cuts) == 1:
                    cuts.append(i - 20)
                elif len(cuts) > 1:
                    if len(cuts) > 2:
                        cuts.remove(cuts[-1])
                    intermediate_cut = []
                    cuts.append(i)
                    intermediate_cut = copy.copy(df[cuts[-2]:cuts[-1]])
                    cuts[-1] = cuts[-2] + intermediate_cut.index(max(intermediate_cut))
                    cuts.append(i)
                else:
                    continue
    return list(dict.fromkeys(cuts[1:]))

def findnth (haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def isNaN(num):
    return num != num

def ocr (image_file):
    image_file0 = image_file[:-4]
    os.chdir(pathog)
    image = cv2.imread(image_file0 + '.jpg')
    os.chdir(pathnew)
    in_dir = (pathnew)   
    
    
       
    config = '--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789'
    configLetters = '--oem 1 --psm 3 tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz'
    
    
    OCR = pytesseract.image_to_string(image, lang='eng', config = configLetters)
    matchers_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 
                'October', 'November', 'December']
    matchers_year = ['2017', '2018', '2019']
    OCR = OCR.replace('\n', ' ')
    OCR = OCR.replace('/', ' ')
    OCR1 = OCR.split(' ')
    try:
        matching_month = [s for s in OCR1 if any(xs in s for xs in matchers_month)][0]
    except IndexError:
        matching_month = ("October")
    try:
        matching_year = [s for s in OCR1 if any(xs in s for xs in matchers_year)][0]
    except IndexError:
        matching_year = ('2017')
    file_name_string = image_file0.split('_')
    state = pathog.split('/')[-1]
    if 'district' in image_file:
        index = file_name_string.index('district')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'county' in image_file:
        index = file_name_string.index('county')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'city' in image_file:
        index = file_name_string.index('city')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'borough' in image_file:
        index = file_name_string.index('borough')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'County' in image_file:
        index = file_name_string.index('County')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'City' in image_file:
        index = file_name_string.index('City')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'Borough' in image_file:
        index = file_name_string.index('Borough')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'District' in image_file:
        index = file_name_string.index('District')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'DISTRICT' in image_file:
        index = file_name_string.index('DISTRICT')
        county = ' '.join(file_name_string[:(index + 1)])
    elif 'menu' in image_file:
        county = ' '.join(file_name_string[:2])
    elif matching_year in image_file:
        index = file_name_string.index(matching_year)
        county = ' '.join(file_name_string[:index])
    else:
        county = file_name_string[0]
    
    if 'lunch' in OCR:
        meal = 'lunch'
    elif 'breakfast' in OCR:
        meal = 'breakfast'
    else:
        meal = "lunch"
    
    preface = (state + ';' + county + ';' + matching_year + ';' + matching_month + ';')
    filename = (in_dir + image_file[:-13] + '.txt')
    headers = ('State;County;Year;Month;Date;Type;Item;Sodium\n')
    totalfile = open(filename, "w+")
    totalfile.write(headers)
    totalfile.close()
    
    number_crop = 40
    for image_file2 in glob(f'*.jpg'):       
        image = cv2.imread(image_file2)
        thresh, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image2 = 255*(image < 128).astype(np.uint8) 
        image2_final = image2[:number_crop,:]
        image_final = image[number_crop:,:]
        
        # See if there is anything
        OCR = pytesseract.image_to_string(image_final, lang='eng', config = configLetters)
        if len(OCR) < 10:
            print("No data: skipped")
            continue
        
        OCR = ''
        length_of_run = max(greatest_line(image)) - 5
        image2_final = image2[:length_of_run,:]
        image_final = image[length_of_run:,:]
        OCR = pytesseract.image_to_string(image2_final, lang='eng', config = config)
        date = copy.copy(OCR)
        preface_interm = (preface + date + ';' + meal + ';')
        OCR = pytesseract.image_to_string(image_final, lang='eng', config = configLetters)
        if "Sodium" in OCR:
            OCR = OCR.replace('\n(', '(')
            OCR = OCR.replace('\n(', '(')
            OCR = OCR.split('\n')
            OCR_new = []
            for i in range(len(OCR)):              
                #if 'mg' in OCR[i] and len(OCR[i]) > 7:
                if len(OCR[i]) > 7:
                    OCR_new.append(OCR[i])
            for i in range(len(OCR_new)):
                if 'mg' in OCR_new[i]:
                    OCR_new[i] = OCR_new[i].replace('(', ';')
                    OCR_new[i] = OCR_new[i].replace('mg', '')
                else:
                    OCR_new[i] = OCR_new[i] + ';'
            OCR_new = '\n'.join(OCR_new)
            OCR_new += '\n'
            OCR = OCR_new
        else:
            OCR = OCR.replace('\n\n','\n')
            OCR += '\n'            
            OCR = OCR.replace('\n',';\n')
            
        OCR = OCR.replace('Sodium','')
        OCR = OCR.replace(')','')
        OCR = OCR.replace('}','')
        OCR = OCR.replace(']','')
        OCR = '\n' + OCR
        OCR = OCR.replace('\n', '\n' + preface_interm)
        OCR = OCR[:OCR.rfind(state)]
        OCR = OCR.replace('+ ','')
        OCR = OCR.replace('« ','')
        OCR = OCR.replace('* ','')
        OCR = OCR.replace('» ','')
        OCR = OCR.replace('+','')
        OCR = OCR.replace('«','')
        OCR = OCR.replace('*','')
        OCR = OCR.replace('»','')
        OCR = OCR.replace('é','')
        OCR = OCR.replace('©','')
    
        OCR = OCR[1:]
    
        test = OCR.split('\n')
        for line in range(len(test)):
            if test[line].count(';') > 7:
                cutindex = findnth(test[line], ';', 7)
                test.insert(line + 1, preface_interm + test[line][cutindex + 1:])
                test[line] = test[line][:cutindex]
        OCR = '\n'.join(test)
        OCR = OCR.encode('utf-8').strip()
        OCR += (b'\n')
        myfile = open(in_dir + '{}.txt'.format(image_file2[:-4]), "w+")
        print(OCR, file=myfile)
        myfile.close()
        
        totalfile2 = open(filename, "ab+")
        totalfile2.write(OCR)
        totalfile2.close()   
    
    
    txt_file = filename
    csv_file = (filename[:-4] + '.csv')
    dataframe = pd.read_csv(txt_file, delimiter=";")
    try:
        
        for rows in range(dataframe.shape[0]):
            if (not isNaN(dataframe['Item'][rows])) and dataframe['Item'][rows][0] == '-':
                dataframe['Item'][rows] = copy.copy(dataframe['Item'][rows][1:])
        try:
            dataframe['Sodium'] = dataframe['Sodium'].replace({'o': '0'}, regex=True)
            dataframe['Sodium'] = dataframe['Sodium'].replace({'O': '0'}, regex=True)
            dataframe['Sodium'] = dataframe['Sodium'].replace({'S': ''}, regex=True)
            dataframe['Sodium'] = dataframe['Sodium'].replace({'f': '1'}, regex=True)
            dataframe['Sodium'] = dataframe['Sodium'].replace({'wi': ''}, regex=True)
            dataframe['Sodium'] = dataframe['Sodium'].replace({'%': '1'}, regex=True)
        except TypeError:
            print("Non Sodium File")   
        #Creates the 'entree' variable using a ML method
        try:
            dataframe.to_csv(csv_file, encoding='utf-8', index=False)
        except UnicodeDecodeError:
            try:
                dataframe.to_csv(csv_file, index=False)
            except UnicodeDecodeError:
                print("Couldn't Create CSV")
    except UnicodeDecodeError:
        print("Couldn't Create CSV")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
real_og_path = os.getcwd()
path = (os.getcwd())
os.chdir(os.getcwd() + '\\images\\')
for state in glob(os.getcwd() + '\\**'):
    pathog = '/'.join(state.split('\\'))
    os.chdir(pathog)
    print(pathog)
    try:
        os.mkdir(path + '\\2017\\' + pathog.split('/')[-1])
    except FileExistsError:
        print("State Folder Exists")
    for image_file in glob(f'*.jpg'):
        os.chdir(pathog)
        print(image_file)
        print("Splitting Rows")
        path = (path + '\\2017\\' + pathog.split('/')[-1] + '\\' + image_file[:-4] + '\\rows\\')
        MISSING_LAST_ROW = True
        
        gray = cv2.imread(image_file)
        gray2 = copy.copy(gray)
        IMG_WIDTH = gray.shape[:2][1]
        IMG_HEIGHT = gray.shape[:2][0]
        for i in range(IMG_WIDTH):
            for j in range(IMG_HEIGHT):
                if not(np.all(gray[j,i] < 1.0)):
                    gray2[j,i] = np.array([255,255,255])
        row_mean = calculate_pixels(gray2)
        #plot_df(row_mean)
        row_cuts = cut_finder(row_mean, 100.0, 50)
        if MISSING_LAST_ROW == True:
            # Add last row
            row_cuts.append(IMG_HEIGHT)
        k = 1
        sorter = '0'
        for i in range(len(row_cuts) - 1):
            cropped = gray[row_cuts[i]:row_cuts[i+1],:]
            if k < 10:
                sorter = '0'
            else:
                sorter = ''
            try:
                os.mkdir('\\'.join(path.split('\\'))[:-6])
            except FileExistsError:
                print("Image folder exists")
            try:
                os.mkdir('\\'.join(path.split('\\')))
            except FileExistsError:
                print("Rows folder exists")
            skimage.io.imsave(path + image_file[:-4] + '_' + sorter + str(k) + '.jpg', cropped)
            k += 1
        
        try:
            os.chdir(path)
            print("Splitting Columns")
            k = 1
            sorter = '0'
            for image_file1 in glob(f'*.jpg'):
                pathnew = (path[:-5])
                MISSING_LAST_COLUMN = True
                gray = cv2.imread(image_file1)
                gray2 = copy.copy(gray)
                thresh, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # was 127, 255; 105, 255 worked alright
                IMG_WIDTH = gray.shape[:2][1]
                IMG_HEIGHT = gray.shape[:2][0]
                length_of_run = max(greatest_line(gray)) - 10
                col_cuts = find_lines(gray, length_of_run = length_of_run, distance = 80)
                if MISSING_LAST_COLUMN == True:
                    # Add last column
                    try:
                        col_cuts.append(int(col_cuts[-1] + ((col_cuts[-1] - col_cuts[0]) / (len(col_cuts) - 1))))
                    except ZeroDivisionError:
                        col_cuts.append(IMG_WIDTH)
                    except IndexError:
                        break
                
                for j in range(len(col_cuts) - 1):
                    if k < 10:
                        sorter = '0'
                    else:
                        sorter = ''
                    cropped = gray2[:, col_cuts[j]:col_cuts[j+1]]
                    skimage.io.imsave(pathnew + image_file1[:-7] + '_' + sorter + str(k) + '.jpg', cropped)
                    k += 1
            print("Running OCR")
            ocr(image_file)
            print('\n')
        except:
            skippedFile = open(real_og_path + "\\skippedFiles.txt", "a+")
            print(image_file, file=skippedFile)
            skippedFile.close()
            print("File Skipped")
            print('\n')
