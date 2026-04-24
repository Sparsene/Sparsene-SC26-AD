#!/bin/bash

if [ -t 1 ]; then
	YELLOW='\033[1;33m'
	NC='\033[0m'
else
	YELLOW=''
	NC=''
fi

printf "%b[NOTE]%b Please follow README.md in the SparseTIR directory.\n" "$YELLOW" "$NC"