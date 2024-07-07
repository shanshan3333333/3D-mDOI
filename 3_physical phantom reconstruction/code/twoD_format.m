function [mua,mus]=twoD_format(twoD_param)
num_type=length(twoD_param);
num=0;
for i=1:num_type
    item=twoD_param{i};
    num_item= length(item);
    for j=1:num_item
        num=num+1;
        mua{num}=item{j}.mua;
        mus{num}=item{j}.mus;
    end
end
