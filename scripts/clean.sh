#!/bin/bash
# CP Clean Script - Removes all cached databases

CP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Cleaning all CP databases..."

# Relay
rm -f "$CP_ROOT/cp_relay.db" && echo "  - Relay DB"

# macOS
rm -rf "$CP_ROOT/apps/macos/src-tauri/.cp" && echo "  - macOS .cp"
rm -rf "$CP_ROOT/.cp" && echo "  - Root .cp"

# iOS Simulator
find ~/Library/Developer/CoreSimulator/Devices -name "mobile_graph.db" -delete 2>/dev/null
find ~/Library/Developer/CoreSimulator/Devices -type d -name ".cp" -exec rm -rf {} + 2>/dev/null
echo "  - iOS Simulator databases"

echo "Done! All databases cleared."
