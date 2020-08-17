%% 
% Final HW solution 
% Students: 
% Alice Eldar 	555863421
% Michal Andelman-Gur 302194287
% Alfredo Lopez G27781827

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data analysis %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;

%% Q1
% read the data file into a table
Kneset_2020a = readtable('Kneset_result_2020a.xlsx');

%% Q2
% a: plot basic election results
% b: bar plot of total number of votes (in counts), per party, over all settlements

votes = Kneset_2020a{:,7:end};
total_votes = sum(votes, 1);
figure;
bar(total_votes);
xlabel('parties');
ylabel('votes (number)');
yline(0.0325*sum(total_votes));

% c: yaxis in log scale
figure;
bar(log(total_votes));
xlabel('parties');
ylabel('votes (log scale)');
yline(log(0.0325*sum(total_votes)));

% d: pie chart of the total number of votes (in percentage), per party, over all settlements
[largest_five, ind_largest_five] = maxk(total_votes, 5);
explode = ismember(total_votes,largest_five);
labels = Kneset_2020a.Properties.VariableNames(7:end);
% labels = labels(ind_largest_five);
figure;
pie(total_votes/sum(total_votes)*100, explode,labels);

% e: summarize some basic voting stats
fprintf('1.	Total registered voters: %d\n',sum(Kneset_2020a{:,3}));
fprintf('2. Total voters: %d\n',sum(Kneset_2020a{:,4}));
fprintf('3. Total voting rate in percentage: %d\n',sum(Kneset_2020a{:,4})/sum(Kneset_2020a{:,3})*100);
fprintf('4. Total valid: %d\nTotal invalid votes: %d\n',sum(Kneset_2020a{:,6}),sum(Kneset_2020a{:,5}));
fprintf('5. Votes threshold: %d %d',3.25,0.0325*sum(Kneset_2020a{:,4}));

%% Q2. 
% top 10 settlements that had the highest percentage of valid votes
percentage_valid_votes = Kneset_2020a{:,6}./Kneset_2020a{:,4}*100;
[top_10_valid_votes, ind_top_10_valid_votes] = maxk(percentage_valid_votes, 10);
[min_10_valid_votes, ind_min_10_valid_votes] = mink(percentage_valid_votes, 10);
top_10_settelments = Kneset_2020a{ind_top_10_valid_votes,1};
low_10_settelments = Kneset_2020a{ind_min_10_valid_votes,1};

%% Q3. ???
% Explore voting pattern correlations
% correlation between the voting pattern in each settlement and the general (total) voting pattern
corr_voting_settlments_vs_general = zeros(size(votes,1),length(total_votes));
general_voting_pattern = total_votes./(size(votes,1));

%% Q4. 
% Cluster the voting data to groups according to voting pattern
% a: samples and features
samples_settelments = Kneset_2020a{:,7:end};
num_samples_settelments = size(samples_settelments,1); 
% Num. of samples (settelments): 1214
num_features_voting = size(samples_settelments,2);
% Num. of features (parties): 30

%% find groups in the data using kmeans clustering algorithm  
% Calculate kmeans for each distance type, 
% using kmeans (k=2:10) with 10 replicates

% distance type: sqeuclidean
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure;
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'sqeuclidean','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx);
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

% distance type: cosine
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'cosine','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx);
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

% distance type: correlation
k_options = 2:10;
silh_avg_all = zeros(size(k_options));
figure
for ii_k = 1:length(k_options)
    rng(0); % For reproducibility
    subplot(5,2,ii_k);
    k = k_options(ii_k);
    [cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
        'correlation','replicates', 10);
    [silh,h] = silhouette(samples_settelments,cidx);
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
% has a different number of citizens (and votes), therefore cosine/correlation
% methods fix this bias.  

%% Q5.
% Using the 'correlation' distance metric
% clustering results with optimal k (k=2) 

%% Q6.
% Plot the clustering results
rng(0); % For reproducibility
% subplot(2,2,i);
k = 2;
[cidx,cmeans] = kmeans(samples_settelments, k,'distance',...
    'correlation','replicates', 10);
% Pie chart showing the percentage of data points in each cluster
figure;
num_cluster1 = sum(cidx==1);
num_cluster2 = sum(cidx==2);
num_clusters = [num_cluster1 num_cluster2];
pie(num_clusters);
labels = {'Cluster 1','Cluster 2'};
legend(labels,'Location','southoutside','Orientation','horizontal');

% stem plot of voting pattern for each cluster
idx_cluster1 = find(cidx==1);
idx_cluster2 = find(cidx==2);
votes_cluster1 = sum(samples_settelments(idx_cluster1,:),1);
votes_cluster2 = sum(samples_settelments(idx_cluster2,:),1);
total_votes = sum(samples_settelments,1);
parties_names = Kneset_2020a.Properties.VariableNames(7:end);
percentage_votes_cluster1 = (votes_cluster1./sum(votes_cluster1))*100;
percentage_votes_cluster2 = (votes_cluster2./sum(votes_cluster2))*100;
percentage_votes = (total_votes./sum(total_votes))*100;

% Plot the figure (are the percentages ok?)
figure;
stem(percentage_votes_cluster1);
hold on
stem(percentage_votes_cluster2);
hold on
plot(percentage_votes,'k');
xticks(1:num_features_voting);
xticklabels(parties_names);
xtickangle(45);
ylabel('voting percentages (%)');
xlabel('parties');
title('voting pattern in each cluster and in the general population','fontsize',14);
legend('Cluster 1','Cluster 2','General voting');

% histogram of correlations values 

% a 3D plot
figure;
plot3(

% f: Select two clusters and try to explain their results using
% the figure you created. What is different between those groups?
% Answer: the two clusters represent right-wing settelments and left-wing
% settelments. It is clear that the right-wing settelemnts tend to vote
% more for "Halikud" and "Yamina", and that left-wing settelments vote more
% for "Kachol-Lavan" and "Haavoda-Meretz". 

%% Q7. 
% the code is generic and runs well on the results of 2019a, 2019b