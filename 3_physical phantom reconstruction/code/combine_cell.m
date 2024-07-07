function output=combine_cell(input)
num_type=length(input);
num=0;
for i=1:num_type
    item=input{i};
    num_item= length(item);
    for j=1:num_item
        num=num+1;
        output{num}=item{j};
    end
end
end