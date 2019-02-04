# Splash-Sensor

Accelerometer (works)-
Accelerometer.ino is our Arduino code that runs the accelerometer. Aside from a few additions that I put in, this is a replica of code that 
you can find on SparkFun. The data that it sends can be viewed on any serial monitor such as the one built into the
arduino IDE, PuTTy, or Python script (what we're doing). 

Serial Logger (works)-
Serial_logger.py is our Python code that looks at the serial port used by the arduino and logs the data into a text file. This was found on
stackexchange.com, however I added a few lines. To use this, you just run the script while the arduino is plugged in. You can see 
the data and after you exit the program (type e in console) the data is saved in a data#.txt file.

Classification (half-baked)-
Classification.py is our Python code that should eventually have all of the classification algorithm inside. Right now, it just separates 
and plots the accelerometer data that's found in the data#.txt files. 
