# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:03:01 2019

@author: Logan
"""
# =============================================================================
# Serial Logger
#
# This script monitors communication on the serial port and logs the data to a
# text file. 
# =============================================================================
import serial
import keyboard  # using module keyboard
import os

#serial connection initialization
serial_port = 'COM4'
baud_rate = 9600 #In arduino, Serial.begin(baud_rate)
ser = serial.Serial(serial_port, baud_rate)


#determines next available file name
i = 0
while os.path.exists("data%s.txt" % i):
    i += 1
write_to_file_path = "data%s.txt" % i

with open(write_to_file_path, "w+") as output_file:
#output_file = open(write_to_file_path, "w+")

    
    while True:
        line = ser.readline()
        line = line.decode("latin-1")  #"utf-8"
        print(line);
        output_file.write(line)
        
        try:  
            if keyboard.is_pressed('e'):  # if key 'q' is pressed 
                print('Program ended')
                break  # finishing the loop
            
        except:
            pass
        
output_file.close()    
ser.close()
