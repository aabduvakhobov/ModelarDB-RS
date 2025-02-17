fn main() {
    cxx_build::bridge("src/lib.rs")
    .std("c++17")
    .include("include")
    .compiler("/usr/bin/clang++")// hardcoded and could be improved in the future 
    .compile("alp_sample_program");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=include/alp_state_values.hpp");
    println!("cargo:rerun-if-changed=include/alp.hpp");    

    println!("cargo:rustc-link-search=.");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/home/abduvoris/Paper-2-Experiments/Paper-2-General/ALP-Integration/alp_sample_program");
    println!("cargo:rustc-link-lib=dylib=ALP");
}