import serial

ser = serial.Serial(port="COM3", baudrate=9600)

try:
    while 1:
        # Ask the user to enter a string.
        to_send = input("Enter a string: ")
        # Send that string to serial port (TX) by encoding
        # it as utf-8 (standard)
        ser.write(to_send.encode())
except Exception as exp:
    print(str(exp))
finally:
    print("Program stopped.")
    ser.close()
