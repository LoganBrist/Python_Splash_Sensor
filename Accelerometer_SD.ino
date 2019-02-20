// SparkFun Code

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

/////////////////////////////////////////////////////////////////////////////////////
void setup(){
  //Pin Assignments
  pinMode (PULL_HIGH, OUTPUT);
  pinMode (GREEN_LED, OUTPUT);
  pinMode (BUTTON,     INPUT);

  //Pull_High Pin
  digitalWrite(PULL_HIGH, HIGH);
  digitalWrite(GREEN_LED, LOW);
  //Intitialze SD writing and accelerometer
  Serial.begin(9600);
  SD.begin(4);
  accel.init();
}
/////////////////////////////////////////////////////////////////////////////////////
//Button_Pushed Function - Returns true if the button is held for the time interval
/////////////////////////////////////////////////////////////////////////////////////

bool Button_Pushed (int pin_number, int interval){
    elapsedMillis timeElapsed;
    while (digitalRead(pin_number) == 1) {
      if (timeElapsed > interval){
        Serial.println("Button Pressed");
        LED_TOGGLE(GREEN_LED);
        while(digitalRead(pin_number) == 1) {}
        Serial.println("Button Released");
        return true;
      }  
    }
    return false;
  }
/////////////////////////////////////////////////////////////////////////////////////
//LED_State Function- Flash LED when changed
/////////////////////////////////////////////////////////////////////////////////////
void LED_TOGGLE (int led_name) {
    //Initial state, to determine ending state and blink pattern
    int State = digitalRead(led_name);
 
    Serial.println(State);
    int n = 0;
    //Blinks
    while(n < 4){  
      digitalWrite(led_name, !State);
      delay(100);
      digitalWrite(led_name, State);
      delay(100);
      n++;
    }
    
    //Toggle value
    digitalWrite(led_name, !State);
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

  //Check if file exists, incrementing number extension each time
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
    
  // Check for button hold to start recording
  while(!Code_Is_Running) {  
    if (Button_Pushed(BUTTON, 50)){
      Code_Is_Running = true;
      Serial.println("Starting data logging");
      dataFile.println("X        Y        Z");
    }
    }
      
  // Log Data
  while(Code_Is_Running) {     
  // Check for new data from the accelerometer and print it out if it's available.
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
      dataFile.close();
      Serial.println(filename + " closed");
      Serial.println();
      Code_Is_Running = false;     
    }
    }
    test_no += 1;
  }   
}

