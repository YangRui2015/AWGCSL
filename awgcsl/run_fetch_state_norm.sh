nohup python3 ppo_train_baseline.py -en FetchReach-v1 -r 0 -g 0 -n 50 -sn True > logs/fetchreach11.txt &
nohup python3 ppo_train_baseline.py -en FetchReach-v1 -r 2 -g 0 -n 50 -sn True > logs/fetchreach12.txt &
nohup python3 ppo_train_baseline.py -en FetchReach-v1 -r 4 -g 1 -n 50 -sn True > logs/fetchreach13.txt &
nohup python3 ppo_train_baseline.py -en FetchReach-v1 -r 6 -g 2 -n 50 -sn True > logs/fetchreach14.txt &
nohup python3 ppo_train_baseline.py -en FetchReach-v1 -r 8 -g 3 -n 50 -sn True > logs/fetchreach15.txt &