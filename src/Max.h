#pragma once

class _MAX
{
public:
    static constexpr unsigned char UINT8 = 0xFF; // 255
    static constexpr unsigned char U8 = 0xFF;
    static constexpr unsigned short UINT16 = 0xFFFF; // 65535
    static constexpr unsigned short U16 = 0xFFFF;
    static constexpr unsigned int UINT32 = 0xFFFFFFFF; // 4294967295
    static constexpr unsigned int U32 = 0xFFFFFFFF;
    static constexpr unsigned long long UINT64 = 0xFFFFFFFFFFFFFFFF; // 18446744073709551615
    static constexpr unsigned long long U64 = 0xFFFFFFFFFFFFFFFF;

    static constexpr char INT8 = 0x7F; // 127
    static constexpr char I8 = 0x7F;
    static constexpr short INT16 = 0x7FFF; // 32767
    static constexpr short I16 = 0x7FFF;
    static constexpr int INT32 = 0x7FFFFFFF; // 2147483647
    static constexpr int I32 = 0x7FFFFFFF;
    static constexpr long long INT64 = 0x7FFFFFFFFFFFFFFF; // 9223372036854775807
    static constexpr long long I64 = 0x7FFFFFFFFFFFFFFF;

    static constexpr float FLOAT = 3.402823e+38F; // FLT_MAX
    static constexpr float F = 3.402823e+38F;
    static constexpr double DOUBLE = 1.7976931348623158e+308; // DBL_MAX
    static constexpr double D = 1.7976931348623158e+308;
};

extern _MAX Max;
