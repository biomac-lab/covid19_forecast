%Inference for the metapopulation SEIR model
clear all, clc

data    = readtable('./bogota_cases.csv');
num_times = size(data,1);

pop       = 8181047;
deaths    = data.death;
incidence = data.confirmed;
num_loc   = 1;

Td   = 9;%average reporting delay
a    = 1.85;%shape parameter of gamma distribution
b    = Td/a;%scale parameter of gamma distribution
rnds = ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers

% observation operator: obs=Hx
H=zeros(num_loc,7*num_loc+8);
for i=1:num_loc
    H(i,(i-1)*7+6)=1;
end

%observation operator: obs=Hx
H_hosp=zeros(num_loc,7*num_loc+8);
for i=1:num_loc
    H_hosp(i,(i-1)*7+7)=1;
end

obs_truth=incidence';
obs_truth_hosp=deaths';

%set OEV
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(25,obs_truth(l,t)^2/100);
    end
    for t=1:num_times
        obs_ave=mean(incidence(l,max(1,t-6):t));
        OEV(l,t)=max(1e-4,obs_ave^2/100);
    end
end

%set OHEV
OHEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OHEV(l,t)=max(25,obs_truth_hosp(l,t)^2/100);
    end
end


num_ens = 300;%number of ensemble
pop0 = double(pop)*ones(1,num_ens);

[x,paramax,paramin]=initialize_SEIHR_deaths(pop0, num_ens, 0, 1:size(pop,1)); %get parameter range

num_var=size(x,1);%number of state variables
%IF setting
Iter=400; % number of iterations

num_para=size(paramax,1); % number of parameters
theta=zeros(num_para,Iter+1); % mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,Iter); %posterior parameters
sig=zeros(1,Iter); %variance shrinking parameter
alp=0.9;           %variance shrinking rate

SIG=(paramax-paramin).^2/4;%initial covariance of parameters

lambda=1.2;%inflation parameter to aviod divergence within each iteration

Sidx   = (1:6:6*num_loc)';
Eidx   = (2:6:6*num_loc)';
Isidx  = (3:6:6*num_loc)';
Iaidx  = (4:6:6*num_loc)';
obsidx = (5:6:6*num_loc)';
obsHidx = (6:6:6*num_loc)';


[x,~,~] = initialize_SEIHR_deaths(pop0,num_ens,0, size(pop,1));

num_forecast = 90;

%Begin looping through observations
num_var = size(x,1);


x_post  = load(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', '400_x_post'));
x_post  = x_post.x_post;

para_post  = load(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', '400_para_post_mean'));
para_post  = para_post.para_post_mean;
theta      = squeeze(mean(para_post,3));


pop      = pop0;

x = checkbound_ini_SEIR_deaths(x,pop0);

M = zeros(num_loc,num_loc,num_times+num_forecast);

display('Starting to simulate')
para_time = para_post;

%Begin looping through observations
x_prior = zeros(num_var,num_ens,num_times+num_forecast);%prior
x_post_sim = zeros(num_var,num_ens,num_times+num_forecast);

pop=pop0;
obs_temp = zeros(num_loc,num_ens,num_times+num_forecast);   % records of reported cases
obs_temp_H = zeros(num_loc,num_ens,num_times+num_forecast); % records of reported hospitalization
recovered_all = zeros(num_loc,num_ens,num_times+num_forecast);
infected_all  = zeros(num_loc,num_ens,num_times+num_forecast);

for t=1:num_times+num_forecast
    
    if t<=num_times
        %para_t  = mean(para_time(:,:,max(t-10,t):t),3);
        para_t = mean(para_time(:,:,:),3);
        paramax = squeeze(max(para_t,[],2));
        paramin = squeeze(min(para_t,[],2));
        SIG     = (paramax-paramin).^2/4;%initial covariance of parameters
        Sigma   = diag( SIG );
        para   = mvnrnd(mean(para_t,2), Sigma,num_ens)'; %generate parameters
        x(end-7:end,:) = para;
    else
        para_t = mean(para_time(:,:,1:num_times),3);
        para_t(1,:) = mean(para_time(1,:,num_times-10:num_times),3);
        paramax = squeeze(max(para_t,[],2));
        paramin = squeeze(min(para_t,[],2));
        
        SIG     = (paramax-paramin).^2/4;%initial covariance of parameters
        Sigma   = diag( SIG );
        para   = mvnrnd(mean(para_t,2), Sigma,num_ens)'; %generate parameters
        x(end-7:end,:) = para;
    end

    % Inflation
    x = mean(x,2)*ones(1,num_ens) + lambda*(x-mean(x,2)*ones(1,num_ens));
    x = checkbound_SEIR_deaths(x,pop);

    if t==1 %<=num_times
        x = x_post(:,:,t);
    end

    %integrate forward
    [x,pop, susceptible, exposed, inf_r, inf_u, inf_d, recovered] = SEIR_deaths(x,M,pop,t,pop0);
    
    recovered_all(1,:,t) = recovered;
    infected_all(1,:,t)  =  inf_r+  inf_u+ inf_d;
    obs_cnt = H*x;%new infection
    obs_cnt_H = H_hosp*x;%new infection
    
    obs_cnt = min(obs_cnt, 1000000);
    obs_cnt_H = min(obs_cnt_H, 20000);
    
    tot_cases_t = round(sum(mean(obs_cnt,2)));
    tot_cases_h = round(sum(mean(obs_cnt_H,2)));
    
    display(strcat('Number of cases in t=', num2str(t),' is=',num2str(tot_cases_t)))
    display(strcat('Number of deaths in t=', num2str(t),' is=',num2str(tot_cases_h)))
    
    %add reporting delay
    for k=1:num_ens
        for l=1:num_loc
            if obs_cnt(l,k)>0
                rnd = datasample(rnds,obs_cnt(l,k));
                for h=1:length(rnd)
                    if (t+rnd(h)<=num_times)
                        obs_temp(l,k,t+rnd(h)) = obs_temp(l,k,t+rnd(h)) + 1;
                    end
                end
            end
        end
    end
    
    obs_temp_H(:,:,t) = obs_cnt_H; % No delay to deaths
    obs_ens   = obs_temp(:,:,t);%observation at t
    obs_ens_H = obs_temp_H(:,:,t);%observation at t
    
    x_prior(:,:,t)=x;%set prior
    
    % AJUST USING DEATHS
    % loop through local observations
    %loop through local observations
    for l=1:num_loc
        %Get the variance of the ensemble
        if t<=num_times
            obs_var = OHEV(l,t);
        else
            obs_var = 0;
        end
        prior_var = var(obs_ens_H(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0%if degenerate
            post_var=1e-3;
            prior_var=1e-3;
        end
        prior_mean = mean(obs_ens_H(l,:));
        if t <= num_times
            post_mean = post_var*(prior_mean/prior_var + obs_truth_hosp(l,t)/obs_var);
        else
            post_mean = post_var*(prior_mean/prior_var); % + obs_truth_hosp(l,t)/obs_var);
        end
           
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens_H(l,:)-prior_mean)-obs_ens_H(l,:);
        %Loop over each state variable (connected to location l)
        rr=zeros(1,num_var);
        neighbors = []; %union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
        neighbors = [neighbors;l];%add location l
        for i=1:length(neighbors)
            idx=neighbors(i);
            
            for j=1:7
                A=cov( x((idx-1)*7+j,:),obs_ens_H(l,:));
                rr((idx-1)*7+j)=A(2,1)/prior_var;
            end
        end
        
        for i=num_loc*7+1:num_loc*7+8
            A=cov(x(i,:),obs_ens_H(l,:));
            rr(i)=A(2,1)/prior_var;
        end
        
        %Get the adjusted variable
        dx = rr'*dy;
        if t<=num_times
            x  = x_post(:,:,t);
        else
            % x  = x+dx;
            % (end-7:end,:) = mean(x_post_sim(end-7:end,:,num_times-7:num_times),3);
        end        %Corrections to DA produced aphysicalities
        x = checkbound_SEIR_deaths(x,pop);
        
    end
    
    % AJUST USING OBSERVATIONS
    % loop through local observations
    %loop through local observations
    for l=1:num_loc
        %Get the variance of the ensemble
        if t<=num_times
            obs_var = OEV(l,t);
        else
            obs_var = 0;
        end
        prior_var = var(obs_ens(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0%if degenerate
            post_var=1e-3;
            prior_var=1e-3;
        end
        prior_mean = mean(obs_ens(l,:));
        if t<=num_times
            post_mean = post_var*(prior_mean/prior_var + obs_truth(l,t)/obs_var);
        else
            post_mean = post_var*(prior_mean/prior_var); %+ obs_truth(l,t)/obs_var);
        end
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
        %Loop over each state variable (connected to location l)
        rr=zeros(1,num_var);
        neighbors = union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
        neighbors = [neighbors;l];%add location l
        
        for i=1:length(neighbors)
            idx=neighbors(i);
            for j=1:7
                A=cov( x((idx-1)*5+j,:),obs_ens(l,:));
                rr((idx-1)*5+j)=A(2,1)/prior_var;
            end
        end
        
        for i=num_loc*7+1:num_loc*7+8
            A=cov(x(i,:),obs_ens(l,:));
            rr(i)=A(2,1)/prior_var;
        end
        
        %Get the adjusted variable
        dx = rr'*dy;
        if t<=num_times
            x  = x+dx;
        else
            x(end-7:end,:) = mean(x_post_sim(end-7:end,:,num_times-10:num_times),3);
        end
        %Corrections to DA produced aphysicalities
        x = checkbound_SEIR_deaths(x,pop);
        
    end
    x_post_sim(:,:,t)=x;
end
x_forecast = x_post_sim;
% where to save the forecast
save(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', 'deaths_post_forecast') , 'obs_temp_H');
save(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', 'xpost_forecast') , 'x_forecast');
save(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', 'recovered') , 'recovered_all');
save(strcat('/Users/chaosdonkey06/Dropbox/BIOMAC/EAKF_Forecast/bogota/checkpoints_agg/', 'infected') , 'infected_all');
