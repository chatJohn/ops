/*
    实现float类型转换为half类型
*/
#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdfloat>
#include <bit>
uint16_t float_to_half(float f){
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (float_bits >> 31) & 0x1;
    uint32_t remain_bits = float_bits & 0x7FFFFFFF;
    if(remain_bits == 0){
        return sign;
    }
    int32_t exp = (remain_bits >> 23) - 127;
    uint32_t mant = (remain_bits & 0x007FFFFF);
    if(exp == 128 && mant != 0){
        return sign | 0x7E00;
    }
    exp += 15;
    if(exp >= 31){
        return sign | 0x7BFF;
    }
    if(exp <= 0){
        if(exp < -10){
            return sign | 0x0001;
        }
        mant |= 0x00800000;
        uint32_t shift = 14 - exp;
        uint32_t a = (1 << (shift - 1)) - 1;
        uint32_t b = (mant >> shift) & 0x1;
        uint32_t rounded_mant = (mant + a + b) >> shift;
        if(rounded_mant == 0){
            return sign | 0x0001;
        }
        return sign | rounded_mant;
    }
    uint32_t rounded = remain_bits + 0x00001000;
    if ((remain_bits & 0x00003FFF) == 0x00001000) { 
        rounded &= ~0x00002000; // Tie-breaker to even
    }
    uint32_t rounded_exp = (rounded >> 23) - 127 + 15;
    uint32_t rounded_mant = (rounded & 0x007FFFFF) >> 13;
    if (rounded_exp >= 31) {
        return sign | 0x7BFF; 
    }
    return sign | (rounded_exp << 10) | rounded_mant;
}
void test(float f) {
    uint16_t bits = float_to_half(f);
    std::float16_t h = std::bit_cast<std::float16_t>(bits);
    std::float16_t ans = static_cast<std::float16_t>(f);
    if(h == ans){
        std::cout << "Test Passed: " << "Float: " << f << "\t -> \tHalf: " << h << "\t -> \tHalf(Hex): 0x" << std::hex << h << std::endl;
    }else{
        throw std::runtime_error("Test Failed: " + std::to_string(f) + " -> " + std::to_string(h) + " != " + std::to_string(ans));
    }
}

int main() {
    // 1. 正常数值
    test(3.14159f);
    test(0.0f);
    // 2. 上溢出
    test(100000.0f);   
    test(-999999.0f);    
    // 3. 下溢出 
    test(1.0e-20f);
    test(-1.0e-30f);
    return 0;
}