#!/bin/bash

rm logs
rm -r ../ModelarDB-data-store/*
cargo build --profile dev-release
RUST_BACKTRACE=1 ./target/dev-release/modelardbd ../ModelarDB-data-store/ 2> logs
