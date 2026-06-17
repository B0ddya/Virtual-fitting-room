#include <SD.h>                      // need to include the SD library
//#define SD_ChipSelectPin 53  //example uses hardware SS pin 53 on Mega2560
#define SD_ChipSelectPin 8  //using digital pin 4 on arduino nano 328, can use other pins
#define button1 2
#define button2 3
#define button3 4
#define button4 5
#define button5 6
#define button6 7
#include <TMRpcm.h>           //  also need to include this library...
#include <SPI.h>

TMRpcm tmrpcm;   // create an object for use in this sketch

unsigned long time = 0;

int i = 0;
String music[7] = {"1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav"};

void setup(){

  tmrpcm.speakerPin = 9; //5,6,11 or 46 on Mega, 9 on Uno, Nano, etc
  //Complimentary Output or Dual Speakers:
  //pinMode(10,OUTPUT); Pin pairs: 9,10 Mega: 5-2,6-7,11-12,46-45 
  
  Serial.begin(9600);
  pinMode(SD_ChipSelectPin,INPUT);
  pinMode(button1,INPUT);
  pinMode(button2,INPUT);
  pinMode(button3,INPUT);
  pinMode(button4,INPUT);
  pinMode(button5,INPUT);
  pinMode(button6,INPUT);
  if (!SD.begin(SD_ChipSelectPin)) {  // see if the card is present and can be initialized:
    Serial.println("SD fail");  
    return;   // don't do anything more if not

  }
  else{   
    Serial.println("SD ok");   
  }
  tmrpcm.play("3.wav");
}



void loop(){  
  if(digitalRead(button1) == HIGH){
    tmrpcm.play("1.wav");
    Serial.println("1"); 
  }
  else if(digitalRead(button2) == HIGH){
    tmrpcm.play("2.wav");
    Serial.println("2"); 
  }
  else if(digitalRead(button3) == HIGH){
    tmrpcm.play("3.wav");
    Serial.println("3"); 
  }
  else if(digitalRead(button4) == HIGH){
    tmrpcm.play("4.wav");
    Serial.println("4"); 
  }
  else if(digitalRead(button5) == HIGH){
    tmrpcm.play("5.wav");
    Serial.println("5"); 
  }
  else if(digitalRead(button6) == HIGH){
    tmrpcm.play("6.wav");
    Serial.println("6"); 
  }
  delay(5);
}
