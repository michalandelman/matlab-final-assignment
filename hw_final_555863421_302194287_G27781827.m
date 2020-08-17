%%
% Final HW solution

% Students:
% Alice Eldar 	555863421
% Michal Andelman-Gur 302194287
% Alfredo Lopez G27781827
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this script processes and explores trends in Israeli election data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

%% Q1
% read 2020 results data to a table
results = readtable('Kneset_result_2020a.xlsx');

%% Q2a
% plot basic election results

raw_votes = table2array(results(:,7:end));
totalvotes_per_party = sum(raw_votes);
totalvotes_all_parties = sum(totalvotes_per_party);
threshold = 3.25;
absolute_thresh = totalvotes_all_parties/100 * threshold;
party_names = string(results.Properties.VariableNames(7:end));
party_names_spaces = strrep(party_names,'_',' ');

figure;

subplot(2,2,1)
bar(totalvotes_per_party);
yline(absolute_thresh);
ticks = 1:length(party_names);
xticks(ticks);
xticklabels(party_names_spaces); 
xtickangle(90);
a = get(gca, 'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',4,'FontWeight','bold');
title('Total votes per party','fontsize',12); 
ylabel('number of votes')

subplot(2,2,2)

bar(totalvotes_per_party);
yline(absolute_thresh);
set(gca, 'YScale', 'log')
ylabel('log number of votes');
party_names = results.Properties.VariableNames(7:end);
ticks = 1:length(party_names);
xticks(ticks);
xticklabels(party_names_spaces);
xtickangle(90);
a = get(gca, 'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',4,'FontWeight','bold');
title('Log total votes per party','fontsize',12);

subplot(2,2,3)

[numbers parties] = maxk(sum(raw_votes),5);
explode = zeros(length(totalvotes_per_party),1)';
explode(parties) = 1;
party_labels_to_remove = find(explode==0);
pie(totalvotes_per_party,explode,party_names_spaces);
p = gca;
pText = flipud(findobj(p,'Type','text'));

for ind = party_labels_to_remove
    pText(ind).String = [];
end

for party_labels = parties
    pText(party_labels).FontSize = 6;
end

hTitle = title('Total share of votes');
set(hTitle, 'FontSize', 12, 'Units', 'normalized', 'Position', [0.5, -0.2, 0]);

% subplot(2,2,4)

basic_voting_stats = table2array(results(:,3:6));
column_totals = sum(basic_voting_stats);
total_registered_voters = column_totals(1);
total_voters = column_totals(2);
total_voting_rate = column_totals(2)/column_totals(1)*100;
total_valid = column_totals(4);
total_invalid = column_totals(3);

str = {sprintf('    Summary statistics:')...
    sprintf('total registered voters: %d', total_registered_voters),...
    sprintf('total voters: %d', total_voters),...
    sprintf('voting rate: %.2f', total_voting_rate),...
    sprintf('valid votes: %d', total_valid),...
    sprintf('invalid votes: %d', total_invalid),...
    sprintf('threshold percent: %.2f', threshold),...
    sprintf('votes needed: %.0f',absolute_thresh)};

annotation('textbox',[0.55 0.05 0.4 0.4]...
    ,'String', str,'FitBoxToText','on')

%% Q2b
% Find the top 10 settlements that had the highest/lowest % valid votes

settlement_names = string(table2array(results(:,1)));
[votes_max settlement_ind] = maxk(basic_voting_stats(:,4),10);

settlement_names_max = settlement_names(settlement_ind);
fprintf('highest number of valid votes:\n');
fprintf('%s\n', settlement_names_max);

[votes_min settlement_ind_min] = mink(basic_voting_stats(:,4),10);
settlement_names_min = settlement_names(settlement_ind_min);
fprintf('lowest number of valid votes:\n');
fprintf('%s\n', settlement_names_min);

%% Q3 voting pattern correlations

% correlation bw voting pattern in each settlement and general voting
% pattern

mean_votes_per_party = mean(raw_votes);
raw_votes_with_mean = [raw_votes;mean_votes_per_party];
correlation_matrix = corr(raw_votes_with_mean');
num_of_settelments_cor_general = 10;

[correlation_max max_sett_index] = maxk(correlation_matrix(end,1:end-1),...
    num_of_settelments_cor_general);
settlement_names_max_corr = settlement_names(max_sett_index);
fprintf('highest correlation to general voting pattern:\n');
fprintf('%s\n', settlement_names_max_corr);

[correlation_min min_sett_index] = mink(correlation_matrix(end,1:end-1),...
    num_of_settelments_cor_general);
settlement_names_min_corr = settlement_names(min_sett_index);
fprintf('lowest correlation to general voting pattern:\n');
fprintf('%s\n', settlement_names_min_corr);

% find the two settlements with highest correlation

mean_removed_matrix = (correlation_matrix(1:end-1,1:end-1));
lower_triangle = tril(mean_removed_matrix,-1);

[max_cor max_index] = maxk(lower_triangle(:),1);
[row, col] = ind2sub(size(mean_removed_matrix), max_index);
settlement_names_max_between = [settlement_names(row); settlement_names(col)];
fprintf('settlements with highest correlation:\n');
fprintf('%s\n', settlement_names_max_between);

[min_corr min_index] = min(lower_triangle(:));
[row, col] = ind2sub(size(mean_removed_matrix), min_index);
settlement_names_min_between = [settlement_names(row); settlement_names(col)];
fprintf('settlements with lowest correlation:\n');
fprintf('%s\n', settlement_names_min_between);

%% Q4
% Cluster the voting data to groups according to voting pattern
% a: samples and features
samples_settelments = raw_votes;
num_samples_settelments = size(samples_settelments,1);
% Num. of samples (settelments): 1214
num_features_voting = size(samples_settelments,2);
% Num. of features (parties): 30

% b-e: find groups in the data using kmeans clustering algorithm
% Calculate kmeans for each distance type,
% using kmeans (k=2:10) with 10 replicates

% kmeans clustering, distance type: sqeuclidean
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure;
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'sqeuclidean','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx,'sqeuclidean');
    silh_avg_all(ii_k) = mean(silh);
    title(sprintf('k = %d',k));
    sgtitle('Silhouette value for different clusters number (distance=sqeuclidean)');
end
subplot(5,2,length(k_options)+1);
hold on
[max_silh,max_silh_IX] = max(silh_avg_all);
% find the optimal k (with maximal silhouette value)
plot(k_options, silh_avg_all)
plot(k_options(max_silh_IX), max_silh,'*r')
txt = sprintf('k = %d',k_options(max_silh_IX));
text(k_options(max_silh_IX)+0.25, max_silh+0.01,txt);
ylabel('Silhouette value');
xlabel('k');
title('Silhouette value per k');
hold off

% kmeans clustering, distance type: cosine
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'cosine','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx,'cosine');
    silh_avg_all(ii_k) = mean(silh);
    title(sprintf('k = %d',k));
    sgtitle('Silhouette value for different clusters number (distance=cosine)');
end
subplot(5,2,length(k_options)+1);
hold on
[max_silh,max_silh_IX] = max(silh_avg_all);
% find the optimal k (with maximal silhouette value)
plot(k_options, silh_avg_all);
plot(k_options(max_silh_IX), max_silh,'*r');
txt = sprintf('k = %d',k_options(max_silh_IX));
text(k_options(max_silh_IX)+0.25, max_silh+0.01,txt);
ylabel('Silhouette value');
xlabel('k');
title('Silhouette value per k');
hold off

% kmeans clustering, distance type: correlation
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'correlation','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx,'correlation');
    silh_avg_all(ii_k) = mean(silh);
    title(sprintf('k = %d',k));
    sgtitle('Silhouette value for different clusters number (distance=correlation)');
end
subplot(5,2,length(k_options)+1);
hold on
[max_silh,max_silh_IX] = max(silh_avg_all);
% find the optimal k (with maximal silhouette value)
plot(k_options, silh_avg_all);
plot(k_options(max_silh_IX), max_silh,'*r');
txt = sprintf('k = %d',k_options(max_silh_IX));
text(k_options(max_silh_IX)+0.25, max_silh+0.01,txt);
ylabel('Silhouette value');
xlabel('k');
title('Silhouette value per k');
hold off

% g: Can you explain why using the 'sqeuclidean' created different results?
% How can you fix it?
% Response: the squared Euclidean distances method gives a large amount of
% weight to the magnitude of the vectors. In our case, each settelment
% has a different number of citizens (and votes) with a big difference
% between them (i.e. Tel-Aviv with 268116 votes vs. Amuka with 97 votes)
% that can bias the results of the Euclidaen distances calculation.
% therefore, cosine/correlation
% methods should be used to fix this bias.

%% Q5.
% clustering results with optimal k (k=5), using 'correlation' distance 

%% Q6.
% Plot the clustering results
rng(0); % For reproducibility
k = 5;
[cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
    'correlation','replicates', 10);

% Pie chart showing the percentage of data points in each cluster
num_clusters = zeros(1,k);
for cluster = 1:k
    num_clusters(:,cluster) = sum(cidx==cluster);
end
figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
subplot(2,2,1);
pie(num_clusters);
colormap([0 0 1; 1 0 0; 1 1 0; 0 1 1; 0 1 0]);
labels = {'Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'};
legend(labels,'Location','northeastoutside','Orientation','vertical');
title('Clusters percentages','Fontsize',14);

% stem plot of voting pattern for each cluster

votes_clusters = zeros(k,num_features_voting);
percentage_votes_cluster = zeros(k,num_features_voting);
total_votes = sum(samples_settelments,1);
percentage_votes = (total_votes./sum(total_votes))*100;
for cluster = 1:k
    idx = find(cidx==cluster);
    votes_clusters(cluster,:) = sum(samples_settelments(idx,:),1);
    percentage_votes_cluster(cluster,:) = ...
        (votes_clusters(cluster,:)./sum(votes_clusters(cluster,:)))*100;
end

% Plot the figure 
subplot(2,2,2);
stem(percentage_votes_cluster(1,:),'b');
hold on
stem(percentage_votes_cluster(2,:),'r');
stem(percentage_votes_cluster(3,:),'y');
stem(percentage_votes_cluster(4,:),'c');
stem(percentage_votes_cluster(5,:),'g');
plot(percentage_votes,'k');
xticks(1:num_features_voting);
xticklabels(party_names_spaces);
xtickangle(90);
ylabel('voting percentages (%)');
xlabel('parties');
title('voting pattern in each cluster and in the general population','fontsize',14);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5',...
    'General voting','Fontsize',8,'Location','northeast','Orientation','vertical');

% d: histogram of the clusters' correlations values
% define indices if each cluster
idx_cluster1 = find(cidx==1);
idx_cluster2 = find(cidx==2);
idx_cluster3 = find(cidx==3);
idx_cluster4 = find(cidx==4);
idx_cluster5 = find(cidx==5);
% find correlation values of each cluster to the total voting pattern
correlations_cluster1 = correlation_matrix(end,idx_cluster1);
correlations_cluster2 = correlation_matrix(end,idx_cluster2);
correlations_cluster3 = correlation_matrix(end,idx_cluster3);
correlations_cluster4 = correlation_matrix(end,idx_cluster4);
correlations_cluster5 = correlation_matrix(end,idx_cluster5);

% plot histogram
subplot(2,2,3);
histogram(correlations_cluster1,'Facecolor','b');
hold on
histogram(correlations_cluster2,'Facecolor','r');
histogram(correlations_cluster3,'Facecolor','y');
histogram(correlations_cluster4,'Facecolor','c');
histogram(correlations_cluster5,'Facecolor','g');
hold off
title('correlation values of each cluster','Fontsize',14);
legend(labels,'Location','northeastoutside','Orientation','vertical');
xlabel('correlation values of each cluster with the general voting pattern');
ylabel('count');

% plot a 3D graph, with the following axes: Number of votes, 
% Voting rate (in percentage), Correlation to the total voting pattern.
num_votes_per_settelment = table2array(results(:,4));
voting_rate_per_settelment = table2array(results(:,4))./table2array(results(:,3))*100;
cor_to_total_voting_pattern = correlation_matrix(end,1:(end-1));
subplot(2,2,4);
plot3(num_votes_per_settelment(idx_cluster1),...
    voting_rate_per_settelment(idx_cluster1),...
    cor_to_total_voting_pattern(idx_cluster1),'o','Color','b');
hold on
plot3(num_votes_per_settelment(idx_cluster2),...
    voting_rate_per_settelment(idx_cluster2),...
    cor_to_total_voting_pattern(idx_cluster2),'o','Color','r');
plot3(num_votes_per_settelment(idx_cluster3),...
    voting_rate_per_settelment(idx_cluster3),...
    cor_to_total_voting_pattern(idx_cluster3),'o','Color','y');
plot3(num_votes_per_settelment(idx_cluster4),...
    voting_rate_per_settelment(idx_cluster4),...
    cor_to_total_voting_pattern(idx_cluster4),'o','Color','c');
plot3(num_votes_per_settelment(idx_cluster5),...
    voting_rate_per_settelment(idx_cluster5),...
    cor_to_total_voting_pattern(idx_cluster5),'o','Color','g');
hold off
xlabel('number of votes per settelment');
ylabel('voting rate per settelment');
zlabel('correlation to total voting pattern');
title('3D plot of the clusters','Fontsize',14);
legend(labels,'Location','northeast','Orientation','vertical');
view(-37,30); %define the camera view

% f: Select two clusters and try to explain their results using
% the figure you created. What is different between those groups?
% Answer: We chose to focus on cluster 1 and 3. The settelements in 
% cluster 1 are half of the total settelments, and are characterized by
% voting mainly to "Kachol Lavan" and "Haavoda-Meretz", with a relatively 
% high correlation with the general voting pattern. Moreover, the voting 
% rate and votes number per settelment are relatively high. In contrary, 
% cluster 3 contains smaller amount of settelments (11%), and is
% characterized by voting mainly to "Hareshima Hameshutefet", with smaller
% correlation with the general voting pattern. The voting rate and votes
% per settelment are smaller compared to cluster 1. 
% Knowing the Israeli geopolitical map, it seems that cluster 3 contains Israeli-Arab
% settelments, while cluster 1 contains central-left-wing Israeli-Jewish
% settelments.

%% Q7.
% the code is generic and runs well on the results of 2019a, 2019b

