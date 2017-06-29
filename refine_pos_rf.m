function [pos, max_response]=refine_pos_rf(im, pos, ferns, app_model, config)
% function [pos, max_response]=refine_pos_rf(im, pos, svm_struct, app_model, config)
max_response=config.max_response;

window_sz=config.window_sz;
app_sz=config.app_sz;

% cell_size=config.features.cell_size;

% config.label_prior_sigma = 15;

[feat, pos_samples, ~, weights]=det_samples(im, pos, window_sz, config.detc);

% Ó¦ÓÃËæ»úÞ§
[hs,probs] = fernsClfApply( feat', ferns);

% scores=svm_struct.w'*feat+svm_struct.b;
% 
% scores=scores.*reshape(weights,1,[]);

% DIY
for i = 1:size(hs,1)
    index = find(probs(:,1)==max(probs(:,1)),1);
    if hs(index)==1
        break;
    end
end

%tpos=round(pos_samples(:, find(scores==max(scores),1)));
tpos=round(pos_samples(:,index));

if isempty(tpos),  return; end

tpos=reshape(tpos,1,[]);

% figure(2), imshow(im),
% hold on, plot(tpos(2), tpos(1), 'xg');

if size(im,3)>1
    im=rgb2gray(im);
end

[~, max_response]=do_correlation(im, tpos, app_sz, [], config, app_model);

% if max_response>1.5*config.max_response && max(scores)>0
% if max_response>config.appearance_thresh && max(scores)>0
if max_response>1.5*config.max_response
    pos=tpos;
else
    max_response=config.max_response;
end