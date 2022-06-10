import struct

def read_binary_fp32(filename):
    with open(filename, "rb") as f:
        byte = f.read(4)
        i = 0
        while byte and i < 100:
            # Do stuff with byte.
            byte = f.read(4)
            # print(f"{byte}")
            print(struct.unpack('f', byte))
            print(byte.hex())
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
    read_binary_fp32("new-fp-light-compressed-20220610-164042.bin")
    #read_binary_int("new-fp-light-compressed-20220610-164042.bin")
