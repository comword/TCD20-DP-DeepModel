#!/usr/bin/env bash
set -e
VIDEO_DIR=`realpath $1`

if [ ! -d $VIDEO_DIR ]; then
	echo "Video directory provided does not exist"
	exit 1
fi

TARGET_DIR="$VIDEO_DIR/processed"
mkdir -p "$TARGET_DIR"

for file in ${VIDEO_DIR}/*.mp4; do
    echo $file
    ext=${file##*.}
    bname=$(basename -- $file)
    fname=${bname%.*}
    split_num=`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${file}`
    printf -v split_num %.0f "$(echo "$split_num / 10" | bc)"
    for i in $(seq 0 $(($split_num-1))); do
        ffmpeg -i ${file} -ss $(($i*10)) -t 10 -an ${TARGET_DIR}/${fname}-${i}.mp4
    done
done