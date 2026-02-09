#!/bin/bash
set -e

# CFS iOS Build Script
# This builds the Rust core as a static library for the iOS simulator.

echo "Building cfs-mobile for iOS Simulator (aarch64)..."
rustup target add aarch64-apple-ios-sim

# Build the library
cargo build -p cfs-mobile --target aarch64-apple-ios-sim --release

echo "Build complete."

# Auto-copy to apps/ios/CFSMobile if it exists
if [ -d "apps/ios/CFSMobile" ]; then
    echo "Updating local copy in apps/ios/CFSMobile/..."
    rsync -av target/aarch64-apple-ios-sim/release/libcfs_mobile.a apps/ios/CFSMobile/
    
    if [ -f "apps/ios/CFSMobile/libcfs_mobile.a" ]; then
        echo "Successfully verified libcfs_mobile.a in apps/ios/CFSMobile/"
        ls -l apps/ios/CFSMobile/libcfs_mobile.a
    else
        echo "ERROR: Failed to find libcfs_mobile.a in target directory after sync!"
        exit 1
    fi
fi

echo "The static library is at: target/aarch64-apple-ios-sim/release/libcfs_mobile.a"
echo ""
echo "Next steps in Xcode:"
echo "1. Open apps/ios/CFSMobile.xcodeproj in Xcode"
echo "2. Run on an iPhone Simulator!"
