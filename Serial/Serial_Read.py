import serial

ser = serial.Serial(port="COM3", baudrate=9600)

try:
    while 1:
        # Listen to the receiver (RX)
        # and read the buffer until /n is received.
        incoming = ser.readline().decode().strip()
        if incoming:
            print("Received from serial: " + incoming)
except Exception as e:
    print("!Error: " + e)
finally:
    print("Program stopped.")
    ser.close()
