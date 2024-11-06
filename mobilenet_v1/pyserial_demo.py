import torch
import torch.nn as nn
import torch.nn.functional as F
import serial       # UART 통신을 시작하기 위해 시리얼 포트 열기
import numpy as np
import struct

# UART 통신 설정
UART_PORT = 'COM4'  # 사용 중인 포트에 맞게 변경 (예: '/dev/ttyUSB0' 또는 'COM3')
BAUD_RATE = 19200   # 보드레이트 설정
TIME_OUT = 0        # 타임아웃 설정

# UART 연결 시도
try:
    ser = serial.Serial(
        UART_PORT,
        BAUD_RATE,
        timeout=TIME_OUT
    )

    # 시리얼포트 접속
    ser.isOpen()

    # 시리얼포트 번호 출력
    print(f"Connected to {ser.name}")

except serial.SerialException as e:
    print(f"Error: Could not open port {UART_PORT}. Reason: {e}")
    exit()  # 포트를 열 수 없으면 프로그램 종료


# 데이터 송신 함수 (python -> FPGA)
# Tensor를 zero-padding -> Tensor 평탄화 (flatten) -> 1D Numpy 배열로 변환 -> binary로 변환

def send_tensor(tensor, data_type):
    # Tensor를 zero-padding
    # 마지막 두 차원에만 패딩 적용 (위,아래, 왼쪽/오른쪽)
    padded_tensor = F.pad(tensor, (1, 1, 1, 1))

    with open('dwcv2_output_tensor.txt', 'w') as f:
        for a in output_tensor.flatten():
            f.write(f"{a.item()}\n")

    # 텐서를 평탄화
    flattened_tensor = padded_tensor.flatten()

    # CPU로 이동 후 data_type으로 변환하여 NumPy 배열로 변환
    flattened_numpy = flattened_tensor.to(data_type).cpu().numpy()

    with open('dwcv2_output_tensor.txt', 'w') as f:
        for a in flattened_numpy:
            f.write(f"{a.item()}\n")

    # Numpy 배열을 4바이트로 변환
    # np.int32(a).tobytes() : a를 4바이트로 변환
    byte_array = [np.int32(flattened_numpy[i]).tobytes() for i in range(len(flattened_numpy))]
    print(byte_array)               # b'\xfd\xff\xff\xff' 이 원소인 array
    #print(len(byte_array[0]))       # 4
    #print(len(byte_array))          # expected : 34*34*32 = 36992

    with open('dwcv2_output_byte.bin', 'wb') as f:
        for byte in byte_array:
            f.write(byte)

    # 송신된 바이트를 비트 배열로 변환 (전체 텐서)
    binary_array = []

    for byte in byte_array:
        binary_data = format(int.from_bytes(byte, byteorder='little', signed=True), '032b')
        binary_array.append(binary_data)

    #print(binary_array)               # '00000000000000000000000000000001'이 원소인 array

    with open('dwcv2_output_binary.bin', 'wb') as f:
        for value in binary_array:
            f.write(value.encode('utf-8'))

    print("변환 완료! 32비트 바이너리 형식으로 저장되었습니다.")

    # UART로 데이터 전송
    for value in binary_array:
        # 바이트 배열을 직접 전송
        ser.write(value.encode('utf-8'))  # utf-32 : 모든 문자를 4바이트(32비트) 고정 길이로 인코딩

'''
# ------------------------------------------------------------------------------------------------------------------------------
# 데이터 수신 함수 (FPGA -> python)
# receive_data 함수는 지정된 크기만큼 데이터를 수신함. 데이터가 완전히 수신될 때까지 반복하여 읽는다.

def receive_data(total_floats):  # ex) total_floats = 1 * 32 * 32 * 32
    # tensor 크기: 1x32x32x32 = 32768개의 float32 값 (32768 * 4 bytes = 131072 bytes)

    total_size = total_floats * 4  # float32는 4바이트
    data = bytearray(total_size)
    bytes_received = 0

    # 데이터가 완전히 수신될 때까지 읽음
    while bytes_received < total_size:
        chunk = ser.read(total_size - bytes_received)  # FPGA -> SW
        if not chunk:
            print("데이터 수신 중 오류 발생")
            break

        data[bytes_received:bytes_received + len(chunk)] = chunk
        bytes_received += len(chunk)

    # 수신된 데이터를 float32로 변환
    floats = []
    for i in range(0, total_size, 4):
        # 4바이트씩 읽어들여서 float32로 변환
        float_value = struct.unpack('f', data[i:i + 4])[0]
        floats.append(float_value)

        # float 리스트를 torch tensor로 변환하고, 원하는 크기로 reshape
        tensor = torch.tensor(floats, dtype=torch.float32).reshape(tensor_shape_tuple)  # reshape() : 변환하고자 하는 텐서의 새로운 차원의 크기를 튜플 형태로 받는다

        return tensor



# 수신된 텐서 데이터 출력
tensor_shape_tuple = (1, 32, 32, 32)
total_tensor_element_num = torch.tensor(tensor_shape_tuple).numel()
tensor_data = receive_data(total_tensor_element_num)
print(tensor_data)
'''


# 검증을 위해 임의의 [1,32,32,32] 모양의 tensor 생성
data_type = torch.float32
output_tensor = torch.randn(1, 32, 32, 32, dtype=data_type)*2-1

# NumPy 배열로 변환
output_array = output_tensor.numpy()

print(output_tensor.size())         # expected : torch.Size([1, 32, 32, 32])

# 데이터 송신
send_tensor(output_tensor, data_type)






