const int pinX = A0;
const int pinY = A1;
const int pinSW = 2;  // пин кнопки
 
void setup() {
  Serial.begin(9600);
  pinMode(pinSW, INPUT_PULLUP); // кнопка с подтяжкой
}
 
void loop() {
  int x = analogRead(pinX);
  int y = analogRead(pinY);
  int button = digitalRead(pinSW); // 0 = нажата, 1 = не нажата
 
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.println(button);
 
  delay(100);
}
