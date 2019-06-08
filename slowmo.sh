# Trim reconstructed video:
echo "Trimming reconstructed video..."
ffmpeg -i video_tennis_HR_gen_gen.avi -t 4 part1.mp4
ffmpeg -i video_tennis_HR_gen_gen.avi -ss 00:00:04 -t 7 slow.mp4
ffmpeg -i video_tennis_HR_gen_gen.avi -ss 00:00:11 part3.mp4
echo "Done."
echo "Slowing down second part..."
ffmpeg -i slow.mp4 -filter:v "setpts=4*PTS" part2.mp4
echo "Done."
echo "Concatenating"
ffmpeg -f concat -i parts.txt -c copy original_slow.mp4
echo "Done."
rm slow.mp4
rm part1.mp4
rm part2.mp4
rm part3.mp4
