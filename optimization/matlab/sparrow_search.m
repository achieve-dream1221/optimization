% 麻雀优化算法
function [Best_pos,Best_fitness ,Iter_curve,History_pos, History_best] = SSA(pop, M,c,d,dim,fobj )
%input
%pop 种群数量
%dim 问题维数
%c 变量上边界最大值
%d 变量下边界最小值
%fobj 适应度函数
%M 最大迭代次数
%output
%Best_pos 最优位置
%Best_fitness 最优适应度值
%Iter_curve 每代最优适应度值
%History_pos 每代种群位置
%History_best 每代最优麻雀位置
%% 参数
P_percent = 0.2; %探索者比例
pNum = round(pop *  P_percent);
lb= c.*ones( 1,dim );    %约束上限
ub= d.*ones( 1,dim );    %约束下限
%% 位置初始化
for i = 1 : pop
    x(i, :) = lb + (ub - lb) .* rand( 1, dim );
    fit(i) = fobj(x(i, :)) ;
end
% 以下找到最小值对应的麻雀群
pFit = fit;
pX = x;
[ Best_fitness, bestI ] = min( fit );
Best_pos = x( bestI, : );
%% 迭代
for t = 1 : M
  [ ans, sortIndex ] = sort(pFit);

  [fmax,B]=max( pFit );
   worse= x(B,:);         %找到最差的个体

   r2=rand(1);
 if(r2<0.8)
    for i = 1 : pNum                                                        % Equation (3)
         r1=rand(1);
        x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M)); %将种群按适应度排序后更新
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %将种群限制在约束范围内
        fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );
    end
  else
  for i = 1 : pNum
  x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
  x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
  fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );
  end
end
 [ fMMin, bestII ] = min( fit );
  bestXX = x( bestII, : );

   for i = ( pNum + 1 ) : pop                     % Equation (4)
         A=floor(rand(1,dim)*2)*2-1;           %产生1和-1的随机数

          if( i>(pop/2))
           x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);
          else
        x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);

         end
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %更新后种群的限制在变量范围
        fit( sortIndex( i ) ) = fobj( x( sortIndex( i ), : ) );                    %更新过后重新计算适应度
   end
  c=randperm(numel(sortIndex));
   b=sortIndex(c(1:20));
  for j =  1  : length(b)      % Equation (5)
    if( pFit( sortIndex( b(j) ) )>(Best_fitness) )
         %如果适应度比最开始最小适应度差的话，就在原来的最好种群上增长一部分值
        x( sortIndex( b(j) ), : )=Best_pos+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -Best_pos)));

        else
        %如果适应度达到开始最小的适应度值，就在原来的最好种群上随机增长或减小一部分
        x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

          end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );

       fit( sortIndex( b(j) ) ) = fobj( x( sortIndex( b(j) ), : ) );
 end
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < Best_fitness )
           Best_fitness= pFit( i );
            Best_pos = pX( i, : );

        end
    end

    Iter_curve(t)=Best_fitness;
    History_pos{t} = x;
    History_best{t} =  Best_pos;
end

function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);

  % Apply the upper bound vector
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move
  s = temp;

