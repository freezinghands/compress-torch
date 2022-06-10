import struct

def read_binary_fp32(filename, byteoffset=0, lines=100):
    with open(filename, "rb") as f:
        byte = f.read(4*byteoffset)
        i = 0
        while byte and i < lines:
            # Do stuff with byte.
            byte = f.read(4)
            # print(f"{byte}")
            print(f"{byte.hex():9}    {struct.unpack('f', byte)}")
            i += 1


def read_binary_int(filename):
    with open(filename, "rb") as f:
        byte = f.read(1)
        i = 0
        while byte and i < 100:
            # Do stuff with byte.
            byte = f.read(1)
            # print(f"{byte}")
            print(int.from_bytes(byte, "little"))
            i += 1

if __name__ == '__main__':
    file_name = "new-fp-light-compressed-20220610-164042.bin"
    byte_offset = 9000
    print(f"Filename: {file_name}")
    print(f"byte offset: {4 * byte_offset}, effectively after {byte_offset} fp32 number.")
    read_binary_fp32(file_name, byteoffset=byte_offset)
    # read_binary_int("new-fp-light-compressed-20220610-164042.bin")
