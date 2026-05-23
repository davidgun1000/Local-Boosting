function [indx]=obtain_index_full_randeffect(num_randeffect,num_param)

count = 1;
for i=1:num_randeffect
    indx(count,1) = i;
    indx(count,2) = i;
    count = count+1;
end

for i=num_randeffect+1:num_randeffect+num_param
    indx(count,1) = i;
    indx(count,2) = i;
    count=count+1;             
end

for i=num_randeffect+1:num_randeffect+num_param
    for j=i+1:num_param+num_randeffect
        indx(count,1) = j;
        indx(count,2) = i;
        count=count+1;         
    end
end

for i=num_randeffect+1:num_randeffect+num_param
    for j=1:num_randeffect
        indx(count,1) = i; 
        indx(count,2) = j;
        count = count+1;
        
    end 
end


% lag=0;
% count=length(indx)+1;
% for i=num_states+1:num_states+num_param
%     for j=i-lag:i
%         indx(count,1)=i;
%         indx(count,2)=j;
%         count=count+1;
%     end
%     lag=lag+1;   
% end



% for i=1:num_param
%     for j=i:num_param
%         indx(count,1) = j;
%         indx(count,2) = i;
%         count=count+1;
%         
%     end
% end



% count=1;
% for i=1:num_states
%     for j=i-1:i
%         indx(count,1)=i;
%         indx(count,2)=j;
%         count=count+1;
%     end
% end
% indx=indx(2:end,:);
% length_indx=length(indx);
% 
% lag=0;
% count=length(indx)+1;
% for i=num_states+1:num_states+num_param
%     for j=i-lag:i
%         indx(count,1)=i;
%         indx(count,2)=j;
%         count=count+1;
%     end
%     lag=lag+1;   
% end

end