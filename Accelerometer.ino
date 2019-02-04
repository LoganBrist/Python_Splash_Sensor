// SparkFun Code

#include <Wire.h> // Library for I2C
#include <SparkFun_MMA8452Q.h> //SFE_MMA8452Q library

// Instance of the MMA8452Q class.
MMA8452Q accel;

// Starts serial and initializes the
//  accelerometer.
void setup()
{
  Serial.begin(9600);
  Serial.println("X        Y        Z");

  accel.init();
}

// Check for new data from the accelerometer and print it out if it's available.
void loop()
{
  if (accel.available())
  {
    accel.read();
    printCalculatedAccels();
    //printOrientation();
    
    Serial.println();
  }
  delay(250); //Log data every .25 seconds
}

// Using the accel.x, accel.y and accel.z variables. Before using these variables you must call accel.read().
void printAccels()
{
  Serial.print(accel.x, 3);
  Serial.print("\t");
  Serial.print(accel.y, 3);
  Serial.print("\t");
  Serial.print(accel.z, 3);
  Serial.print("\t");
}

// Using the accel.cx, accel.cy, and accel.cz variables.
void printCalculatedAccels()
{ 
  Serial.print(accel.cx, 3);
  Serial.print("\t");
  Serial.print(accel.cy, 3);
  Serial.print("\t");
  Serial.print(accel.cz, 3);
  Serial.print("\t");
}


// Using the accel.readPL() function, which reads the portrait/landscape status of the sensor.
void printOrientation()
{
  byte pl = accel.readPL();
  switch (pl)
  {
  case PORTRAIT_U:
    Serial.print("Portrait Up");
    break;
  case PORTRAIT_D:
    Serial.print("Portrait Down");
    break;
  case LANDSCAPE_R:
    Serial.print("Landscape Right");
    break;
  case LANDSCAPE_L:
    Serial.print("Landscape Left");
    break;
  case LOCKOUT:
    Serial.print("Flat");
    break;
  }
}
