import torch
import torch.nn as nn
import torch.nn.functional as F
import serial       # UART 통신을 시작하기 위해 시리얼 포트 열기
import numpy as np
import struct

# ------------------------------------------------------------------------------------------------------------------------------
# UART PORT OPEN 함수
def uart_setup():
    # UART 통신 설정
    UART_PORT = 'COM4'  # 사용 중인 포트에 맞게 변경 (예: '/dev/ttyUSB0' 또는 'COM3')
    BAUD_RATE = 19200   # 보드레이트 설정
    TIME_OUT = 1        # 타임아웃 설정

    # UART 연결 시도
    try:
        ser = serial.Serial(
            UART_PORT,
            BAUD_RATE,
            timeout=TIME_OUT,
            bytesize=serial.EIGHTBITS,  # 데이터 비트 수: 8 비트
            parity=serial.PARITY_NONE,  # 패리티 비트: 없음
            stopbits=serial.STOPBITS_ONE  # 스톱 비트: 1 비트
        )
        # 시리얼포트 접속
        ser.isOpen()
        # 시리얼포트 번호 출력
        print(f"Connected to {ser.name}\n")
        return ser

    except serial.SerialException as e:
        print(f"Error: Could not open port {UART_PORT}. Reason: {e}")
        exit()  # 포트를 열 수 없으면 프로그램 종료


# ------------------------------------------------------------------------------------------------------------------------------
# weight 바이너리 문자열 데이터 송신 함수
def send_weight(ser, pth):
    # .bin 파일을 열어서 한 줄씩 읽기
    with open(f'{pth}', 'r') as f:  # 파일 경로는 실제 경로로 수정
        num_of_byte_sent = 0

        for line in f:
            # 줄 끝의 \n을 제거하고 바이트로 변환하여 전송
            weight = line.strip()               # .strip()을 사용해 \n과 공백 제거
            byte_data = weight.encode('utf-8')  # 문자열을 바이트로 인코딩     # str 타입의 문자열을 utf-8 방식으로 인코딩해서 바이트로 변환한다
            ser.write(byte_data)                # 바이트 데이터 전송
            ser.flush()  # 송신 버퍼가 비워질 때까지 대기
            num_of_byte_sent += 1       # 전송한 횟수 누적

    print("weight 데이터 송신 완료! >_<")
    print(f"number of transmissions : {num_of_byte_sent}\n")

# ------------------------------------------------------------------------------------------------------------------------------
# 데이터 송신 함수 (python -> FPGA)
# Tensor를 zero-padding -> Tensor 평탄화 (flatten) -> float32 수치 형식의 1D Numpy 배열로 변환
# -> 바이트 배열 byte array로 변환 -> 비트 문자열 binary array로 변환 -> 비트 문자열을 다시 바이트로 변환한 후 UART 전송을 해야 합니다.
def send_tensor(ser, tensor, data_type):
    # Tensor를 zero-padding
    # 마지막 두 차원에만 패딩 적용 (위,아래, 왼쪽/오른쪽)
    padded_tensor = F.pad(tensor, (1, 1, 1, 1))
    padded_tensor = padded_tensor.type(data_type)

    # 텐서를 평탄화
    # CPU로 이동 후 data_type으로 변환하여 NumPy 배열로 변환
    flattened_numpy = padded_tensor.flatten().to(data_type).cpu().numpy()

    # float32 값을 4바이트씩 변환하여 byte array 생성 (bytes 원소 1개 크기는 4byte)
    byte_array = [np.float32(flattened_numpy[i]).tobytes() for i in range(len(flattened_numpy))]

    #print(byte_array)       #   b'\x9f_\x86\xbf' 이 원소인 array

    # 송신된 바이트를 비트 문자열 binary_array 로 변환
    binary_array = []
    for byte in byte_array:
        binary_data = format(int.from_bytes(byte, byteorder='big', signed=False), '032b')
        binary_array.append(binary_data)

    # 비트 문자열을 바이트로 변환하여 UART 전송
    num_of_byte_sent = 0

    for binary_str in binary_array:
        # 비트 문자열을 바이트로 변환
        byte_to_send = int(binary_str, 2).to_bytes(4, byteorder='big')  # 4바이트로 변환
        ser.write(byte_to_send)
        num_of_byte_sent += 1

    print("output tensor 송신 완료! >_<")
    print(f"number of transmissions : {num_of_byte_sent}\n")

# ------------------------------------------------------------------------------------------------------------------------------
# 데이터 수신 함수 (FPGA -> python)
# receive_data 함수는 지정된 크기만큼 데이터를 수신함. 데이터가 완전히 수신될 때까지 반복하여 읽는다.
def receive_data(ser, tensor_shape_tuple, data_type):
    # 텐서의 총 원소 수 계산: 1x32x32x32 = 32768개
    total_floats = np.prod(tensor_shape_tuple)  # NumPy를 사용하여 원소 수 계산

    # total_size : 총 받아야 하는 바이트 개수
    total_bytes = total_floats * 4  # float32는 4바이트
    print(f"총 받아야 하는 바이트 개수 : {total_bytes}")              # 32768 * 4 bytes = 131072 bytes

    data = bytearray(total_bytes)

    # bytes_received : 현재 받은 바이트 개수
    bytes_received = 0

    # 받아야 할 데이터를 모두 수신할 때까지 읽음
    while bytes_received < total_bytes:
        chunk = ser.read(total_bytes - bytes_received)  # FPGA -> SW
        if not chunk:
            print("데이터 수신 중 오류 발생")
            break

        data[bytes_received:bytes_received + len(chunk)] = chunk
        bytes_received += len(chunk)

    # 오류 처리
    if bytes_received != total_bytes:
        print(f"수신된 데이터의 크기가 예상과 다릅니다. 예상 크기: {total_bytes}, 실제 수신 크기: {bytes_received}")
        return None

    # 수신된 데이터를 float32로 변환
    floats = []
    for i in range(0, total_bytes, 4):
        float_value = struct.unpack('f', data[i:i + 4])[0]
        floats.append(float_value)

    # 텐서를 reshape하기 위해 shape tuple 사용
    tensor = torch.tensor(floats, dtype=data_type).reshape(tensor_shape_tuple)
    return tensor


# weight 데이터 송신 검증
ser = uart_setup()          # UART PORT OPEN
ser.write(b'10101000')      # SEND INSTRUCTION (type : bytes)
pth='weight_binary_files/fp32/dwcv2_weight_bin.bin'
send_weight(ser, pth)         # SEND WEIGHT BINARY STRING DATA

# output tensor 데이터 송신 검증
data_type = torch.float32
output_tensor = torch.randn(1, 32, 32, 32, dtype=data_type)*2-1     # 검증을 위해 임의의 [1,32,32,32] 모양의 tensor 생성
send_tensor(ser, output_tensor, data_type)      # 데이터 송신

# binary 데이터 수신 검증
tensor_shape_tuple = (1, 32, 32, 32)    # 수신될 텐서 형태 정의
tensor_data = receive_data(ser, tensor_shape_tuple, data_type)      # 데이터 수신
print(tensor_data)      # 수신된 tensor 결과 출력
print(tensor_data.size())      # 수신된 tensor 크기 출력