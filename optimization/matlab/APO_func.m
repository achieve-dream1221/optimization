%https://www.mathworks.com/matlabcentral/fileexchange/162656-artificial-protozoa-optimizer
% Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization
function [bestProtozoa,bestFit,recordtime] = APO_func(fhd,dim,pop_size,iter_max,Xmin,Xmax,varargin)
% random seeds
    stm = RandStream('swb2712','Seed',sum(100*clock));
    RandStream.setGlobalStream(stm);
% global best
    targetbest = [300;400;600;800;900;1800;2000;2200;2300;2400;2600;2700];
    Fidvec = cell2mat(varargin);
    Fid = Fidvec(1);
    runid = Fidvec(2);
    name_convergence_curve = ['APO_Fid_',num2str(Fid),'_',num2str(dim),'D','.dat'];
    f_out_convergence = fopen(name_convergence_curve,'a');

    ps = pop_size;  % ps denotes protozoa size
    np = 1;         % np denotes neighbor pairs     np_max can be set in floor((ps-1)/2)
    pf_max = 0.1;   % pf_max denotes proportion fraction maximum

% set points to plot convergence_curve
  if  runid ==1
        for i=1:51 % 51 points to plot
            if i==1
                iteration=1;
                fprintf(f_out_convergence,'%s:%s\t','iter_F',num2str(Fid));
            else
                iteration=iter_max/50*(i-1);
            end
            fprintf(f_out_convergence,'%d\t',iteration);
        end
        fprintf(f_out_convergence,'\n');
  end
%
tic;
protozoa=zeros(ps,dim);    % protozoa
newprotozoa=zeros(ps,dim); % new protozoa
epn=zeros(np,dim); % epn denotes effect of paired neighbors
% initilization
for i = 1:ps
    protozoa(i,:) = Xmin + rand(1,dim).*(Xmax-Xmin);
end
% evaluate fitness value
protozoa_Fit = feval(fhd,protozoa',varargin{:});
% find the bestProtozoa and bestFit
[bestval,bestid] = min(protozoa_Fit);
bestProtozoa = protozoa(bestid,:);  % bestProtozoa
bestFit = bestval; % bestFit
     fprintf(f_out_convergence,'%s\t%.15f\t',num2str(runid),bestFit-targetbest(Fid));
%%  Main loop
for iter=2:iter_max
    [protozoa_Fit,index] = sort(protozoa_Fit);
    protozoa= protozoa(index,:);
    pf = pf_max*rand; % proportion fraction
    ri=randperm(ps,ceil(ps*pf)); % rank index of protozoa in dormancy or reproduction forms
    for i=1:ps
        if ismember(i,ri) %  protozoa is in dormancy or reproduction form
           pdr=1/2*(1+cos((1-i/ps)*pi)); % probability of dormancy and reproduction
           if rand<pdr  % dormancy form
                newprotozoa(i,:)=  Xmin + rand(1,dim).*(Xmax-Xmin);
           else  % reproduction form
                flag=[1,-1];  % +- (plus minus)
                Flag=flag(ceil(2*rand));
                Mr=zeros(1,dim); % Mr is a mapping vector in reproduction
                Mr(1,randperm(dim,ceil(rand*dim)))=1;
                newprotozoa(i,:)= protozoa(i,:) + Flag*rand*(Xmin+rand(1,dim).*(Xmax-Xmin)).*Mr;
           end
        else  % protozoa is foraging form
           f= rand*(1+cos(iter/iter_max*pi)); % foraging factor
           Mf=zeros(1,dim);  % Mf is a mapping vector in foraging
           Mf(1,randperm(dim,ceil(dim*i/ps)))=1;
           pah= 1/2*(1+cos(iter/iter_max*pi)); % probability of autotroph and heterotroph
           if rand<pah  % protozoa is in autotroph form
            j= randperm(ps,1); % j denotes the jth randomly selected protozoa
            for k=1:np % np denotes neighbor pairs
              if i==1
                 km=i; % km denotes the k- (k minus)
                 kp=i+randperm(ps-i,1); % kp denotes the k+ (k plus)
              elseif i==ps
                 km=randperm(ps-1,1);
                 kp=i;
              else
                 km=randperm(i-1,1);
                 kp=i+randperm(ps-i,1);
              end
              % wa denotes weight factor in the autotroph forms
              wa=exp(-abs(protozoa_Fit(1,km)/(protozoa_Fit(1,kp)+eps)));
              % epn denotes effect of paired neighbors
              epn(k,:)=wa*(protozoa(km,:)-protozoa(kp,:));
            end
            newprotozoa(i,:)= protozoa(i,:)+ f*(protozoa(j,:)-protozoa(i,:)+1/np*sum(epn,1)).*Mf;

         else   % protozoa is in heterotroph form
            for k=1:np % np denotes neighbor pairs
             if i==1
                 imk=i;   % imk denotes i-k (i minus k)
                 ipk=i+k; % ipk denotes i+k (i plus k)
             elseif i==ps
                 imk=ps-k;
                 ipk =i;
             else
                 imk=i-k;
                 ipk=i+k;
             end
             % neighbor limit range in [1,ps]
             if  imk<1
                    imk=1;
             elseif ipk>ps
                    ipk=ps;
             end
             % denotes weight factor in the heterotroph form
             wh=exp(-abs(protozoa_Fit(1,imk)/(protozoa_Fit(1,ipk)+eps)));
             epn(k,:)=wh*(protozoa(imk,:)-protozoa(ipk,:));
            end
             flag=[1,-1];  % +- (plus minus)
             Flag=flag(ceil(2*rand));
             Xnear=(1+Flag*rand(1,dim)*(1-iter/iter_max)).* protozoa(i,:);
             newprotozoa(i,:)=protozoa(i,:)+f*(Xnear-protozoa(i,:)+1/np*sum(epn,1)).*Mf;
          end
       end
   end
       newprotozoa = ((newprotozoa>=Xmin)&(newprotozoa<=Xmax)).*newprotozoa...
                     +(newprotozoa<Xmin).*Xmin+(newprotozoa>Xmax).*Xmax;
       newprotozoa_Fit= feval(fhd,newprotozoa',varargin{:});
       bin = (protozoa_Fit > newprotozoa_Fit)';
       protozoa(bin==1,:) = newprotozoa(bin==1,:);
       protozoa_Fit(bin==1) = newprotozoa_Fit(bin==1);
       [bestFit,bestid] = min(protozoa_Fit);
       bestProtozoa = protozoa(bestid,:);
       if mod(iter,iter_max/50)==0
             fprintf(f_out_convergence,'%.15f\t',bestFit-targetbest(Fid));
       end
 end
    recordtime = toc;
    fprintf(f_out_convergence,'\n');
    fclose(f_out_convergence);
end