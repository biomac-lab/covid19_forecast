function [x,paramax,paramin] = initialize_SEIHR(pop,num_ens,M,seeds)
%Initialize the metapopulation SEIHRS model
num_loc=size(pop,1);
% num_var=5*num_loc+6;
% S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
% prior range

Slow=1.0;Sup=1.0;%susceptible fraction
Elow=0;Eup=0;%exposed
Irlow=0;Irup=0;%documented infection
Iulow=0;Iuup=0;%undocumented infection
Idlow=0;Idup=0;%undocumented infection

obslow=0;obsup=0; %reported case
obsHlow=0;obsHup=0; %reported hospitalizations

betalow=0.8;betaup=1.5; %transmission rate
mulow=0.4;muup=0.55; %relative transmissibility
thetalow=1;thetaup=1.75; %movement factor
Zlow=3;Zup=4;  %latency period
alphalow=0.02;alphaup=0.5; %reporting rate
Dlow=3; Dup=4;  %infectious period
Llow=10; Lup=14; %hospitalization period
IFRlow = 0.005; IFRup=0.02; %hospitalization period

num_state_var = 7;

%range of model state including variables and parameters
xmin=[];
xmax=[];

for i=1:num_loc
    xmin = [xmin; Slow*pop(i); Elow*pop(i); Irlow*pop(i); Iulow*pop(i); Idlow*pop(i);  obslow; obsHlow];
    xmax = [xmax; Sup*pop(i); Eup*pop(i);   Irup*pop(i); Iuup*pop(i);  Idup*pop(i); obsup; obsHup];
end

xmin = [xmin; betalow; mulow;  thetalow; Zlow; alphalow; Dlow; Llow; IFRlow];
xmax = [xmax; betaup;  muup;   thetaup;  Zup;  alphaup;  Dup; Lup; IFRup];
paramax = xmax(end-7:end);
paramin = xmin(end-7:end);

%seeding in cities with more than 100 cases
seeds_id = seeds;

for i =1:numel(seeds_id)
    seedidi = seeds_id(i);
    %E
    xmin((seedidi-1)*num_state_var+2)=0; xmax((seedidi-1)*num_state_var+2)=3;
    %Is
    xmin((seedidi-1)*num_state_var+3)=0; xmax((seedidi-1)*num_state_var+3)=1;
    %Ia
    xmin((seedidi-1)*num_state_var+4)=0; xmax((seedidi-1)*num_state_var+4)=0;
    xmin((seedidi-1)*num_state_var+5)=0; xmax((seedidi-1)*num_state_var+5)=0;

    %H
    xmin((seedidi-1)*num_state_var+6)=0; xmax((seedidi-1)*num_state_var+6)=0;
    xmin((seedidi-1)*num_state_var+7)=0; xmax((seedidi-1)*num_state_var+7)=0;

end

%Latin Hypercubic Sampling
x=lhsu(xmin,xmax,num_ens);
x=x';
for i=1:num_loc
    x((i-1)*num_state_var+1:(i-1)*num_state_var+7,:)=round(x((i-1)*num_state_var+1:(i-1)*num_state_var+7,:));
end


function s=lhsu(xmin,xmax,nsample)
nvar=length(xmin);
ran=rand(nsample,nvar);
s=zeros(nsample,nvar);
for j=1: nvar
    idx = randperm(nsample);
    P = (idx'-ran(:,j))/nsample;
    s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
end