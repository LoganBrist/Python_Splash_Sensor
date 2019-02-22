// SparkFun Code

---------------------------------------------------------------------------------------
//   Libraries and global variables


#include <SparkFun_MMA8452Q.h> //SFE_MMA8452Q library
#include <SPI.h> // Library for SPI (communication with SD)
#include <SD.h>  // SD card functions
#include <elapsedMillis.h>

// Instance of the MMA8452Q class.
MMA8452Q accel;

//Pin names.
File dataFile;
const uint8_t BUTTON         = 3;
const uint8_t PULL_HIGH      = 2;
const uint8_t GREEN_LED      = 5;
     uint64_t Blink_Counter  = 0; 
     
---------------------------------------------------------------------------------------
//   Functions

///////////////////////////////////////////////////////////////////////////////////////
// Initilize SD card function
///////////////////////////////////////////////////////////////////////////////////////
void start_SD(){
    bool SD_success = SD.begin(4);
    if (SD_success)
        Serial.println("SD card initialized successful...");
    else {
        Serial.println("SD card not recognized... Insert card and then hit resert button (program stopping)");
        while(1){
          //blink light
          Blink_Counter++;
          bool state = digitalRead(GREEN_LED);
          if (Blink_Counter % 10000 == 0)
              state = !state;
              
          digitalWrite(GREEN_LED,state);
            if (Button_Pushed (BUTTON, 50)) {
              LED_TOGGLE(GREEN_LED,LOW);
              start_SD();
              break;
              }
            }
        }
}

/////////////////////////////////////////////////////////////////////////////////////
//Button_Pushed Function - Returns true if the button is held for the time interval
/////////////////////////////////////////////////////////////////////////////////////

bool Button_Pushed (int pin_number, int interval){
    elapsedMillis timeElapsed;
    while (digitalRead(pin_number) == 1) {
      if (timeElapsed > interval){
        while(digitalRead(pin_number) == 1) {}
        return true;
      }  
    }
    return false;
  }
/////////////////////////////////////////////////////////////////////////////////////
//LED_State Function- Flash LED when changed
/////////////////////////////////////////////////////////////////////////////////////
void LED_TOGGLE (int led_name, bool EndState) {
    //Initial state, to determine ending state and blink pattern
    //digitalRead(led_name);

    int n = 0;
    //Blinks
    while(n < 4){  
      digitalWrite(led_name, EndState);
      delay(100);
      digitalWrite(led_name, !EndState);
      delay(100);
      n++;
    }
    
    //Toggle value
    digitalWrite(led_name, EndState);
}


-------------------------------------------------------------------------------------
// MAIN

/////////////////////////////////////////////////////////////////////////////////////
// Setup function
/////////////////////////////////////////////////////////////////////////////////////
void setup(){
  //Pin Assignments
  pinMode (PULL_HIGH, OUTPUT);
  pinMode (GREEN_LED, OUTPUT);
  pinMode (BUTTON,     INPUT);

  //Pull_High Pin
  digitalWrite(PULL_HIGH, HIGH);
  digitalWrite(GREEN_LED, LOW);
  
  //Intitialze serial communication
  Serial.begin(9600);
  Serial.println("Serial connection initialized successfully...");

  //Initialize Accelerometer
  accel.init();

  Serial.println("Ready to log data.");
}
/////////////////////////////////////////////////////////////////////////////////////
// Data Collection loop
/////////////////////////////////////////////////////////////////////////////////////

void loop(){
  int test_no = 0;
  while(1){
  bool Code_Is_Running = false;
  
   //Strings for filename and test number
  String Test_no;
  String filename;

  // Hold program until button press
  while(!Code_Is_Running) {  
    if (Button_Pushed(BUTTON, 50)){
      LED_TOGGLE(GREEN_LED,HIGH);
      Code_Is_Running = true;
    }
   }

  //Initialize SD
  start_SD();
  digitalWrite(GREEN_LED,HIGH); //Button toggle signal is put here to overwrite the blink
  
  //Check for an open filename
  while(true){
      Test_no  = String(test_no);
      filename = "data" + Test_no + ".txt"; 
      if (SD.exists(filename) == false)
          break;
      test_no++;
    }
  
  //Open file
  Serial.println(filename + " opened");
  dataFile = SD.open(filename, FILE_WRITE); 
  Serial.println("Starting data logging");
  dataFile.println("X        Y        Z");
    
     
  // Log Data loop
  while(Code_Is_Running) {     
    //Write accelerometer data to SD card
    if (accel.available()){
      accel.read();
      dataFile.print(accel.cx, 3);
      dataFile.print("\t");
      dataFile.print(accel.cy, 3);
      dataFile.print("\t");
      dataFile.print(accel.cz, 3);
      dataFile.print("\t");
      dataFile.println();    
      } 

    // Check for button hold to stop recording
    if (Button_Pushed(BUTTON, 50)){
      LED_TOGGLE(GREEN_LED,LOW);
      dataFile.close();
      Serial.println(filename + " closed");
      Serial.println();
      Code_Is_Running = false;     
      } 
    }
    test_no += 1;
  }   
}
