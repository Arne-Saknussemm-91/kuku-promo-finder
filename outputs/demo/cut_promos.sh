#!/usr/bin/env bash
# Cuts every ranked promo from episode videos (set VIDEO dir / pattern as needed).
set -euo pipefail
mkdir -p promos

ffmpeg -y -loglevel error -ss 30.41 -to 64.29 -i "videos/episode-001.mp4" -c:v libx264 -preset veryfast -c:a aac "promos/promo_01_E01-S1_part1.mp4"
mv "promos/promo_01_E01-S1_part1.mp4" "promos/promo_01_E01-S1.mp4"

ffmpeg -y -loglevel error -ss 28.55 -to 64.02 -i "videos/episode-002.mp4" -c:v libx264 -preset veryfast -c:a aac "promos/promo_02_E02-S1_part1.mp4"
mv "promos/promo_02_E02-S1_part1.mp4" "promos/promo_02_E02-S1.mp4"

ffmpeg -y -loglevel error -ss 7.04 -to 28.55 -i "videos/episode-002.mp4" -c:v libx264 -preset veryfast -c:a aac "promos/promo_03_E02-S2_part1.mp4"
mv "promos/promo_03_E02-S2_part1.mp4" "promos/promo_03_E02-S2.mp4"
