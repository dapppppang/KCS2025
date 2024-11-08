import serial
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, filename='transmission_log.txt', filemode='w')

ser = serial.Serial('COM3', 19200)
data = b'10101010101010101010101010101010'

# 데이터 로깅
logging.debug(f"전송할 데이터: {data}")
bytes_sent = ser.write(data)
logging.debug(f"전송된 바이트 수: {bytes_sent}")
