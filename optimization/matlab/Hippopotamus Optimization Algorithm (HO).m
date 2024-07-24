%https://blog.csdn.net/weixin_46204734/article/details/136414953
%https://www.mathworks.com/matlabcentral/fileexchange/160088-hippopotamus-optimization-algorithm-ho
%% Designed and Developed by Mohammad Hussien Amiri and Nastaran Mehrabi Hashjin
function[Best_score,Best_pos,HO_curve]=HO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness)
lowerbound=ones(1,dimension).*(lowerbound);                              % Lower limit for variables
upperbound=ones(1,dimension).*(upperbound);                              % Upper limit for variables
%% Initialization
for i=1:dimension
    X(:,i) = lowerbound(i)+rand(SearchAgents,1).*(upperbound(i) - lowerbound(i));                          % Initial population
end
for i =1:SearchAgents
    L=X(i,:);
    fit(i)=fitness(L);
end
%% Main Loop
for t=1:Max_iterations
    %% Update the Best Condidate Solution
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);                                  % Optimal location
        fbest=best;                                           % The optimization objective function
    elseif best<fbest
        fbest=best;
        Xbest=X(location,:);
    end
    for i=1:SearchAgents/2
        %% Phase1: The hippopotamuses position update in the river or pond (Exploration)
        Dominant_hippopotamus=Xbest;
        I1=randi([1,2],1,1);
        I2=randi([1,2],1,1);
        Ip1=randi([0,1],1,2);
        RandGroupNumber=randperm(SearchAgents,1);
        RandGroup=randperm(SearchAgents,RandGroupNumber);
        % Mean of Random Group
        MeanGroup=mean(X(RandGroup,:)).*(length(RandGroup)~=1)+X(RandGroup(1,1),:)*(length(RandGroup)==1);
        Alfa{1,:}=(I2*rand(1,dimension)+(~Ip1(1)));
        Alfa{2,:}= 2*rand(1,dimension)-1;
        Alfa{3,:}= rand(1,dimension);
        Alfa{4,:}= (I1*rand(1,dimension)+(~Ip1(2)));
        Alfa{5,:}=rand;
        A=Alfa{randi([1,5],1,1),:};
        B=Alfa{randi([1,5],1,1),:};
        X_P1(i,:)=X(i,:)+rand(1,1).*(Dominant_hippopotamus-I1.*X(i,:));
        T=exp(-t/Max_iterations);
        if T>0.6
            X_P2(i,:)=X(i,:)+A.*(Dominant_hippopotamus-I2.*MeanGroup);
        else
            if rand()>0.5
                X_P2(i,:)=X(i,:)+B.*(MeanGroup-Dominant_hippopotamus);
            else
                X_P2(i,:)=((upperbound-lowerbound)*rand+lowerbound);
            end
        end
        X_P2(i,:) = min(max(X_P2(i,:),lowerbound),upperbound);
        L=X_P1(i,:);
        F_P1(i)=fitness(L);
        if(F_P1(i)<fit(i))
            X(i,:) = X_P1(i,:);
            fit(i) = F_P1(i);
        end
        L2=X_P2(i,:);
        F_P2(i)=fitness(L2);
        if(F_P2(i)<fit(i))
            X(i,:) = X_P2(i,:);
            fit(i) = F_P2(i);
        end
    end
    %% Phase 2: Hippopotamus defense against predators (Exploration)
    for i=1+SearchAgents/2 :SearchAgents
        predator=lowerbound+rand(1,dimension).*(upperbound-lowerbound);
        L=predator;
        F_HL=fitness(L);
        distance2Leader=abs(predator-X(i,:));
        b=unifrnd(2,4,[1 1]);
        c=unifrnd(1,1.5,[1 1]);
        d=unifrnd(2,3,[1 1]);
        l=unifrnd(-2*pi,2*pi,[1 1]);
        RL=0.05*levy(SearchAgents,dimension,1.5);
        if fit(i)> F_HL
            X_P3(i,:)=RL(i,:).*predator+(b./(c-d*cos(l))).*(1./distance2Leader);
        else
            X_P3(i,:)=RL(i,:).*predator+(b./(c-d*cos(l))).*(1./(2.*distance2Leader+rand(1,dimension)));
        end
        X_P3(i,:) = min(max(X_P3(i,:),lowerbound),upperbound);
        L=X_P3(i,:);
        F_P3(i)=fitness(L);
        if(F_P3(i)<fit(i))
            X(i,:) = X_P3(i,:);
            fit(i) = F_P3(i);
        end
    end
    %% Phase 3: Hippopotamus Escaping from the Predator (Exploitation)
    for i=1:SearchAgents
        LO_LOCAL=(lowerbound./t);
        HI_LOCAL=(upperbound./t);
        Alfa{1,:}= 2*rand(1,dimension)-1;
        Alfa{2,:}= rand(1,1);
        Alfa{3,:}=randn;
        D=Alfa{randi([1,3],1,1),:};
        X_P4(i,:)=X(i,:)+(rand(1,1)).*(LO_LOCAL+D.* (HI_LOCAL-LO_LOCAL));
        X_P4(i,:) = min(max(X_P4(i,:),lowerbound),upperbound);
        L=X_P4(i,:);
        F_P4(i)=fitness(L);
        if(F_P4(i)<fit(i))
            X(i,:) = X_P4(i,:);
            fit(i) = F_P4(i);
        end
    end
    best_so_far(t)=fbest;
    disp(['Iteration ' num2str(t) ': Best Cost = ' num2str(best_so_far(t))]);
    Best_score=fbest;
    Best_pos=Xbest;
    HO_curve=best_so_far;
end
end

% The code has been taken from the study:
%'Multiobjective cuckoo search for design optimization Xin-She Yang, Suash Deb'.
% Coded by Hemanth Manjunatha on Nov 13 2015.

% Input parameters
% n     -> Number of steps
% m     -> Number of Dimensions
% beta  -> Power law index  % Note: 1 < beta < 2
% Output
% z     -> 'n' levy steps in 'm' dimension
%_________________________________________________________________________
function [z] = levy(n,m,beta)
    num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator

    den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator
    sigma_u = (num/den)^(1/beta);% Standard deviation
    u = random('Normal',0,sigma_u,n,m);

    v = random('Normal',0,1,n,m);
    z =u./(abs(v).^(1/beta));

  end