#!/bin/sh

## 5-way experiments
# 5-way 1-shot -> 5-way 1-shot (euclidean)
#python3 main.py -N 5 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 21566 &

# 5-way 1-shot -> 5-way 1-shot (cosine)
#python3 main.py -N 5 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 21567 &

# 5-way 1-shot -> 5-way 5-shot (euclidean)
#python3 main.py -N 5 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 21568 &

# 5-way 1-shot -> 5-way 5-shot (cosine)
#python3 main.py -N 5 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 21569 &

# 5-way 5-shot -> 5-way 1-shot (euclidean)
python3 main.py -N 5 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 21570 &

# 5-way 5-shot -> 5-way 1-shot (cosine)
python3 main.py -N 5 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 21571 &

# 5-way 5-shot -> 5-way 5-shot (euclidean)
#python3 main.py -N 5 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 21572 &

# 5-way 5-shot -> 5-way 5-shot (cosine)
python3 main.py -N 5 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 21573 &


## 10-way experiments
# 10-way 1-shot -> 5-way 1-shot (euclidean)
python3 main.py -N 10 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 31655 &

# 10-way 1-shot -> 5-way 1-shot (cosine)
python3 main.py -N 10 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 31656 &

# 10-way 1-shot -> 5-way 5-shot (euclidean)
python3 main.py -N 10 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 31657 &

# 10-way 1-shot -> 5-way 5-shot (cosine)
python3 main.py -N 10 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 31658 &

# 10-way 5-shot -> 5-way 1-shot (euclidean)
python3 main.py -N 10 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 31659 &

# 10-way 5-shot -> 5-way 1-shot (cosine)
python3 main.py -N 10 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 31660 &

# 10-way 5-shot -> 5-way 5-shot (euclidean)
python3 main.py -N 10 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 31661 &

# 10-way 5-shot -> 5-way 5-shot (cosine)
python3 main.py -N 10 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 31662 &


## 20-way experiments
# 20-way 1-shot -> 5-way 1-shot (euclidean)
python3 main.py -N 20 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 25273 &

# 20-way 1-shot -> 5-way 1-shot (cosine)
python3 main.py -N 20 -K 1 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 25274 &

# 20-way 1-shot -> 5-way 5-shot (euclidean)
python3 main.py -N 20 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 25275 &

# 20-way 1-shot -> 5-way 5-shot (cosine)
python3 main.py -N 20 -K 1 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 25276 &

# 20-way 5-shot -> 5-way 1-shot (euclidean)
python3 main.py -N 20 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --tb_port 25277 &

# 20-way 5-shot -> 5-way 1-shot (cosine)
python3 main.py -N 20 -K 5 -Q 15 -Nt 5 -Kt 1 -Qt 15 --dot --tb_port 25278 &

# 20-way 5-shot -> 5-way 5-shot (euclidean)
#python3 main.py -N 20 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --tb_port 25279 &

# 20-way 5-shot -> 5-way 5-shot (cosine)
#python3 main.py -N 20 -K 5 -Q 15 -Nt 5 -Kt 5 -Qt 15 --dot --tb_port 25280 &
