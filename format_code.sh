#!/bin/bash

files=(`ls *.h *.c`)

for f in ${files[@]}
do
  temp=$(mktemp)
  #expand -t 4 "$f" > "$temp" && mv "$temp" "$f"
  #unexpand -t 4 --first-only "$f" | expand -t 2 > "$temp" && mv "$temp" "$f"
  awk 'BEGIN { inside = 0 }
{
    # 1. Xoá comment đa dòng và đơn dòng
    if (inside) {
        if (sub(/^.*\*\//, "")) inside = 0; else next;
    }
    gsub(/\/\*.*\*\//, "");
    if (sub(/\/\*.*$/, "")) inside = 1;
    sub(/\/\/.*$/, "");

    # 2. Xoá khoảng trắng thừa ở cuối dòng
    sub(/[[:space:]]+$/, "");

    # 3. Thu gọn nhiều dòng trống thành 1
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
