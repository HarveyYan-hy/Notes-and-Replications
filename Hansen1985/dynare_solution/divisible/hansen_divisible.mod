//----------------------------------------------------------------------
// Variables
//----------------------------------------------------------------------
var
y // output
c // consumption
k // capital
h // labor
inv // investment
lambda; // tfp


varexo eps;




//----------------------------------------------------------------------
// Parameters
//----------------------------------------------------------------------
parameters
theta // capital share
delta 
beta 
gamma // lambda AR(1)
A // h in utility
sigma_eps
;







//----------------------------------------------------------------------
// Parameter values
//----------------------------------------------------------------------
theta = 0.36;
delta = 0.025;
beta = 0.99;
A = 2;
gamma = 0.95;
sigma_eps=0.00712;





//----------------------------------------------------------------------
// Model
//----------------------------------------------------------------------
model;
y = lambda * k(-1)^theta * h^(1-theta); // production function
k = (1-delta)*k(-1) + inv; // capital accumulation
y - c - inv = 0; // market clear

A / (1-h) = (1/c) * lambda * k(-1)^(theta) * (1-theta) * h^(-theta); // foc
1/c = (beta/c(+1)) * (1-delta + lambda(+1)* theta* k^(theta-1)* h(+1)^(1-theta));

log(lambda) = gamma*log(lambda(-1)) + eps;

end;




//----------------------------------------------------------------------
// Steady State Model
//----------------------------------------------------------------------
steady_state_model;
lambda = 1;
h = (1+(A/(1-theta))*(1 - (beta*delta*theta)/(1-beta*(1-delta))))^(-1);
k = h*((1/beta -(1-delta))/(theta*lambda))^(1/(theta-1));
y = lambda * k^theta * h^(1-theta); 
inv = delta*k;
c = y - inv;
end;






//----------------------------------------------------------------------
// Steady state + checks
//----------------------------------------------------------------------
steady;
check;
model_info;





//----------------------------------------------------------------------
// simulation
//----------------------------------------------------------------------

% Save figures
options_.graph_format = 'pdf';
options_.TeX = 0;


shocks;
var eps; stderr sigma_eps;
end;



stoch_simul(order=1,irf=20,hp_filter=1600,simul_replic=100,periods=115) y c inv k h;





//----------------------------------------------------------------------
// table1
//----------------------------------------------------------------------


% read out simulation
simulated_series_raw=get_simul_replications(M_,options_);

%filter series
simulated_series_filtered=NaN(size(simulated_series_raw));
for ii=1:options_.simul_replic
    [trend, cycle]=sample_hp_filter(simulated_series_raw(:,:,ii)',1600);
    simulated_series_filtered(:,:,ii)=cycle';
end

%get variable positions
y_pos=strmatch('y',M_.endo_names,'exact');
c_pos=strmatch('c',M_.endo_names,'exact');
i_pos=strmatch('inv',M_.endo_names,'exact');
k_pos=strmatch('k',M_.endo_names,'exact');
h_pos=strmatch('h',M_.endo_names,'exact');


var_positions=[y_pos; c_pos; i_pos; k_pos; h_pos];
%get variable names
var_names=M_.endo_names_long(var_positions,:);

%Compute standard deviations
std_mat=std(simulated_series_filtered(var_positions,:,:),0,2)*100;

%Compute correlations
for ii=1:options_.simul_replic
    corr_mat(1,ii)=corr(simulated_series_filtered(y_pos,:,ii)',simulated_series_filtered(y_pos,:,ii)');
    corr_mat(2,ii)=corr(simulated_series_filtered(y_pos,:,ii)',simulated_series_filtered(c_pos,:,ii)');
    corr_mat(3,ii)=corr(simulated_series_filtered(y_pos,:,ii)',simulated_series_filtered(i_pos,:,ii)');
    corr_mat(5,ii)=corr(simulated_series_filtered(y_pos,:,ii)',simulated_series_filtered(h_pos,:,ii)');
    
    % 资本要错位一期
    corr_mat(4,ii)=corr(simulated_series_filtered(y_pos,2:end,ii)', simulated_series_filtered(k_pos,1:end-1,ii)');
end

%Print table with results
title_string='Economy with divisble labor'
fprintf('\n%-40s \n',title_string)
fprintf('%-20s \t %11s \t %11s \n','','std(x)','corr(y,x)')
for ii=1:size(corr_mat,1)
    fprintf('%-20s \t %3.2f (%3.2f) \t %3.2f (%3.2f) \n',var_names{ii,:},mean(std_mat(ii,:,:),3),std(std_mat(ii,:,:),0,3),mean(corr_mat(ii,:),2),std(corr_mat(ii,:),0,2))
end








