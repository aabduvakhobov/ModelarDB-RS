fn main() {
    cxx_build::bridge("src/lib.rs")
    .std("c++17")
    .include("include")
    .compiler("/usr/bin/clang++")// hardcoded and could be improved in the future 
    .compile("alp_cc");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/alp_state_values.hpp");
    println!("cargo:rerun-if-changed=include/alp.hpp");    

    println!("cargo:rustc-link-search=/home/abduvoris/ModelarDB-Home/ModelarDB-RS/crates/alp_cc/");
    println!("cargo:rustc-link-lib=dylib=ALP");
}