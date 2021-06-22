
function x = checkbound_ini_SEIHR(x,pop)
%S,E,Is,Ia,H,obs,...,beta,mu,theta,Z,alpha,D
betalow=0.8;betaup=1.5; %transmission rate
mulow=0.4;muup=0.6; %relative transmissibility
thetalow=1;thetaup=1.75; %movement factor
Zlow=3;Zup=4;  %latency period
alphalow=0.02;alphaup=1.0; %reporting rate
Dlow=3; Dup=4;  %infectious period
Llow=8; Lup=14; %hospitalization period
IFRlow=0.1/100; IFRup=2/100; %hospitalization period

xmin=[betalow;mulow;thetalow;Zlow;alphalow;Dlow;Llow;IFRlow];
xmax=[betaup;muup;thetaup;Zup;alphaup;Dup;Lup;IFRup];

num_loc=size(pop,1);
num_state_var = 7;

for i=1:num_loc
    %S
    x((i-1)*num_state_var+1,x((i-1)*num_state_var+1,:)<0)=0;
    x((i-1)*num_state_var+1,x((i-1)*num_state_var+1,:)>pop(i,:))=pop(i,x((i-1)*num_state_var+1,:)>pop(i,:));
    %E
    x((i-1)*num_state_var+2,x((i-1)*num_state_var+2,:)<0)=0;
    %Ir
    x((i-1)*num_state_var+3,x((i-1)*num_state_var+3,:)<0)=0;
    %Iu
    x((i-1)*num_state_var+4,x((i-1)*num_state_var+4,:)<0)=0;
    %Id
    x((i-1)*num_state_var+4,x((i-1)*num_state_var+5,:)<0)=0;
    %obs cases
    x((i-1)*num_state_var+6,x((i-1)*num_state_var+6,:)<0)=0;
    %obs hospitalizations
    x((i-1)*num_state_var+7,x((i-1)*num_state_var+7,:)<0)=0;
end
for i=1:8
    temp=x(end-8+i,:);
    index=(temp<xmin(i))|(temp>xmax(i));
    index_out=find(index>0);
    index_in=find(index==0);
    %redistribute out bound ensemble members
    x(end-8+i,index_out)=datasample(x(end-8+i,index_in),length(index_out));
end