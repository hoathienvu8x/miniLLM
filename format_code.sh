#!/bin/bash

files=(`ls *.h *.c`)

for f in ${files[@]}
do
  temp=$(mktemp)
  expand -t 4 "$f" > "$temp" && mv "$temp" "$f"
  unexpand -t 4 --first-only "$f" | expand -t 2 > "$temp" && mv "$temp" "$f"
  #cat -s "$f" > "$temp"
  #sed 's/[[:space:]]*$//' "$f" | cat -s > "$temp"
  #mv "$temp" "$f"
  continue
  awk '{ 
  sub(/[[:space:]]+$/, "");
  if (NF > 0) { 
    print; 
    blank = 0; 
  } else {
    if (blank == 0) { 
      print ""; 
      blank = 1; 
    } 
  } 
}' "$f" > "$temp"
mv "$temp" "$f"
done
