#!/bin/sh
set -e

echo "=== Updating agent lattice packages ==="
(cd agent && pnpm up_lattice)

echo ""
echo "=== Updating ai_web lattice packages ==="
(cd ai_web && pnpm up_lattice)

echo ""
echo "=== Done ==="
