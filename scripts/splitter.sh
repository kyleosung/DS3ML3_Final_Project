tail -n +2 Jan_Data_Parsed.csv | split -l 100000 - DataTrain/Chess_Jan_
for file in DataTrain/Chess_Jan_*
do
    head -n 1 Jan_Data_Parsed.csv > tmp_file
    cat $file >> tmp_file
    mv -f tmp_file $file
done