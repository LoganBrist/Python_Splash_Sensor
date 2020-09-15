# Splash-Sensor

Accelerometer-
Accelerometer.ino is the Arduino code that runs the accelerometer. This is derived of code that you can find on SparkFun. The data it sends can be viewed on any serial monitor such as the one built into the arduino IDE, PuTTy,
or Python script (what we're doing). 

Serial Logger -
Serial_logger.py is the Python code that looks at the serial port used by the arduino and logs the data into a text file. This was based on guidance from stackexchange.com. To use this, you just run the script while the arduino is plugged in. You can see 
the data being printed and after you exit the program (type e in console) the X Y Z acceleration data is saved in a data#.txt file.

Classification -
Classification.py is the Python code that has all of the classification algorithm inside. It separates 
and plots the accelerometer data that's found in the data#.txt files. 

Main.py- Python code that wraps together all critical lines of code for analyzing the accelerometer data. All functions called in main.py are found in CustomFunctions.py. Main.py holds code to load, format, filter, and plot data. It also has code for statistical analysis, feature extraction, and machine learning. this file is meant to be kept neat and straight forward. 

CustomFunctions.py - Holds all referenced code in main.py. Code that includes loops, many labels, etc. is placed here to keep main.py clean.
