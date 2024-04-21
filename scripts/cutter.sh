cut -d ',' -f 6,7,15,16,17,18,19,25,29,35,41 ../Data/lichess_db_standard_rated_2019-01.csv > Jan_Data_Parsed.csv

# ( ) 1.	game_id 
# ( ) 2.	type 
# ( ) 3.	result 
# ( ) 4.	white_player 
# ( ) 5.	black_player 
# (X) 6.	white_elo 
# (X) 7.	black_elo 
# ( ) 8.	time_control 
# ( ) 9.	num_ply 
# ( ) 10.	termination
# ( ) 11.	white_won 
# ( ) 12.	black_won 
# ( ) 13.	no_winner 
# ( ) 14.	move_ply 
# (X) 15.	move 
# (X) 16.	cp 
# (X) 17.	cp_rel 
# (X) 18.	cp_loss 
# (X) 19.	is_blunder_cp 
# ( ) 20.	winrate 
# ( ) 21.	winrate_elo 
# ( ) 22.	winrate_loss 
# ( ) 23.	is_blunder_wr 
# ( ) 24.	opp_winrate 
# (X) 25.	white_active 
# ( ) 26.	active_elo
# ( ) 27.	opponent_elo 
# ( ) 28.	active_won 
# (X) 29.	is_capture 
# ( ) 30.	clock 
# ( ) 31.	opp_clock 
# ( ) 32.	clock_percent 
# ( ) 33.	opp_clock_percent 
# ( ) 34.	low_time 
# (X) 35.	board 
# ( ) 36.	active_bishop_count
# ( ) 37.	active_knight_count 
# ( ) 38.	active_pawn_count 
# ( ) 39.	active_queen_count 
# ( ) 40.	active_rook_count 
# (X) 41.	is_check 
# ( ) 42.	num_legal_moves 
# ( ) 43.	opp_bishop_count 
# ( ) 44.	opp_knight_count 
# ( ) 45.	opp_pawn_count 
# ( ) 46.	opp_queen_count 
# ( ) 47.	opp_rook_count

