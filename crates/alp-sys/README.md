Compile instructions:
- Run: `clang++ -std=c++17 -o main main.cpp -I./include -L. -lALP`
- Run: `LD_LIBRARY_PATH=. ./main`

Compile ALP for release instructions:
cmake -B . -S .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang 


ALP-sys (Rust ALP binding) issues:
- correct type to store: ffor_base, exceptions, exceptions_count (maybe deduced from exceptions list), exceptions_positions

# ALP segment components
- Exceptions: Vec<f32>
- Exceptions positions: Vec<u16>
- Exceptions count (can be extracted from previous vectors) 
- Bit width: u8
- Base arr: i32
- Encoded int (bit packed integers)
- Factor: u8
- Exponent: u8