% ---------------------------------------------------------------------------------------------------------------------- %
%  Authors, Inventors, and Programmers: Dr. Sowmya R, Dr. M. Premkumar, and Dr. Pradeep Jangir
%  E-Mail: mprem.me@gmail.com
% ---------------------------------------------------------------------------------------------------------------------- %
% ---------------------------------------------------------------------------------------------------------------------- %
% Please Refer the Following:
% Newton-Raphson-based optimizer: A new population-based metaheuristic algorithm for continuous optimization problems
% Engineering Applications of Artificial Intelligence,
% Volume 128, 2024,107532.
% ISSN 0952-1976
% https://doi.org/10.1016/j.engappai.2023.107532.
% https://www.sciencedirect.com/science/article/pii/S0952197623017165)
% Cite As: R. Sowmya, M. Premkumar, and P. Jangir, “Newton-Raphson-Based Optimizer: A New Population-Based Metaheuristic Algorithm for Continuous Optimization Problems,”
% Engineering Applications of Artificial Intelligence, Vol. 128, pp. 107532, February 2024.
% Website: https://premkumarmanoharan.wixsite.com/mysite/downloads
% ---------------------------------------------------------------------------------------------------------------------- %
%https://www.mathworks.com/matlabcentral/fileexchange/158066-newton-raphson-based-optimizer-nrbo?s_tid=prof_contriblnk
function [Best_Score, Best_Pos, CG_curve] = NRBO(N, MaxIt, LB, UB, dim, fobj)
    % Input arguments:
    % N     - Number of particles in the population
    % MaxIt - Maximum number of iterations
    % LB    - Lower bound of the search space
    % UB    - Upper bound of the search space
    % dim   - Dimensionality of the search space
    % fobj  - Objective function to minimize/maximize
    % Deciding Factor for Trap Avoidance Operator
    DF = 0.6;
    % Initialize the bounds for each dimension
    LB = ones(1, dim) * LB;
    UB = ones(1, dim) * UB;
    % Initialization of the population
    Position = initialization(N, dim, UB, LB);
    Fitness = zeros(N, 1); % Vector to store individual costs
    % Calculate the initial fitness for each particle
    for i = 1:N
        Fitness(i) = fobj(Position(i,:));
    end
    % Determine the best and worst fitness in the initial population
    [~, Ind] = sort(Fitness);
    Best_Score = Fitness(Ind(1));
    Best_Pos = Position(Ind(1),:);
    Worst_Cost = Fitness(Ind(end));
    Worst_Pos = Position(Ind(end),:);
    % Initialize convergence curve
    CG_curve = zeros(1, MaxIt);
    % Main optimization loop
    for it = 1:MaxIt
        % Dynamic parameter delta, decreases over iterations
        delta = (1 - ((2 * it) / MaxIt)) .^ 5;
        % Loop over all particles in the population
        for i = 1:N
            % Randomly select two different indices for differential evolution
            P1 = randperm(N, 2);
            a1 = P1(1); a2 = P1(2);
            % Calculate the step size rho
            rho = rand * (Best_Pos - Position(i,:)) + rand * (Position(a1,:) - Position(a2,:));
            % Apply Newton-Raphson Search Rule
            Flag = 1;
            NRSR = SearchRule(Best_Pos, Worst_Pos, Position(i,:), rho, Flag);
            X1 = Position(i,:) - NRSR + rho;
            X2 = Best_Pos - NRSR + rho;
            % Update position of particle
            Xupdate = zeros(1, dim);
            for j = 1:dim
                X3 = Position(i,j) - delta * (X2(j) - X1(j));
                a1 = rand; a2 = rand;
                Xupdate(j) = a1 * (a1 * X1(j) + (1 - a2) * X2(j)) + (1 - a2) * X3;
            end
            % Trap Avoidance Operator to prevent local optima
            if rand < DF
                theta1 = -1 + 2 * rand(); theta2 = -0.5 + rand();
                beta = rand < 0.5;
                u1 = beta * 3 * rand + (1 - beta); u2 = beta * rand + (1 - beta);
                if u1 < 0.5
                    X_TAO = Xupdate +  theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                else
                    X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position(i,:)) + theta2 * delta * (u1 * mean(Position) - u2 * Position(i,:));
                end
                Xnew = X_TAO;
            else
                Xnew = Xupdate;
            end
            % Enforce boundary conditions
            Xnew = min(max(Xnew, LB), UB);
            % Evaluate new solution
            Xnew_Cost = fobj(Xnew);
            % Update the best and worst positions
            if Xnew_Cost < Fitness(i)
                Position(i,:) = Xnew;
                Fitness(i) = Xnew_Cost;
                % Update the global best solution
                if Fitness(i) < Best_Score
                    Best_Pos = Position(i,:);
                    Best_Score = Fitness(i);
                end
            end
            % Update the global worst solution
            if Fitness(i) > Worst_Cost
                Worst_Pos = Position(i,:);
                Worst_Cost = Fitness(i);
            end
        end
        % Update convergence curve
        CG_curve(it) = Best_Score;
        % Display iteration information
        disp(['Iteration ' num2str(it) ': Best Fitness = ' num2str(CG_curve(it))]);
    end
end


% ---------------------------------------------------------------------------------------------------------------------- %
%  Authors, Inventors, and Programmers: Dr. Sowmya R, Dr. M. Premkumar, and Dr. Pradeep Jangir
%  E-Mail: mprem.me@gmail.com
% ---------------------------------------------------------------------------------------------------------------------- %
% ---------------------------------------------------------------------------------------------------------------------- %
% Please Refer the Following:
% Newton-Raphson-based optimizer: A new population-based metaheuristic algorithm for continuous optimization problems
% Engineering Applications of Artificial Intelligence,
% Volume 128, 2024,107532.
% ISSN 0952-1976
% https://doi.org/10.1016/j.engappai.2023.107532.
% https://www.sciencedirect.com/science/article/pii/S0952197623017165)
% Cite As: R. Sowmya, M. Premkumar, and P. Jangir, “Newton-Raphson-Based Optimizer: A New Population-Based Metaheuristic Algorithm for Continuous Optimization Problems,”
% Engineering Applications of Artificial Intelligence, Vol. 128, pp. 107532, February 2024.
% Website: https://premkumarmanoharan.wixsite.com/mysite/downloads
% ---------------------------------------------------------------------------------------------------------------------- %
% Newton-Raphson Search Rule Function
function NRSR = SearchRule(Best_Pos, Worst_Pos, Position, rho, Flag)
    % Inputs:
    % Best_Pos, Worst_Pos   - Best and worst positions in the population
    % Position              - Current position
    % rho                   - Step size
    % Flag                  - Indicator for search rule application
    dim = size(Position, 2); % Number of dimensions
    DelX = rand(1, dim) .* abs(Best_Pos - Position); % Delta X for search rule
    % Initial Newton-Raphson step
    NRSR = randn * ((Best_Pos - Worst_Pos) .* DelX) ./ (2 * (Best_Pos + Worst_Pos - 2 * Position));
    % Adjust position based on flag
    if Flag == 1
        Xa = Position - NRSR + rho;
    else
        Xa = Best_Pos - NRSR + rho;
    end
    % Further refine the Newton-Raphson step
    r1 = rand; r2 = rand;
    yp = r1 * (mean(Xa + Position) + r1 * DelX);
    yq = r2 * (mean(Xa + Position) - r2 * DelX);
    NRSR = randn * ((yp - yq) .* DelX) ./ (2 * (yp + yq - 2 * Position));
end