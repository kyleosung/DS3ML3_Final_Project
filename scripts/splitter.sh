mkdir ../Data/DataTrain
tail -n +2 ../Data/Jan_Data_Parsed.csv | split -l 100000 - ../Data/DataTrain/Chess_Jan_
for file in ../Data/DataTrain/Chess_Jan_*
do
    head -n 1 ../Data/Jan_Data_Parsed.csv > ../Data/DataTrain/tmp_file
    cat $file >> ../Data/DataTrain/tmp_file
    mv -f ../Data/DataTrain/tmp_file $file
done
