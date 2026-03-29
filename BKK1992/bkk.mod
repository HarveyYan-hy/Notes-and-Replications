//======================================================================
// BKK (1992) Two-Country RBC with Time-to-Build (Dynare)
//======================================================================

/*
Z - Inventories (stock): inventory available in period t is Z_{t-1}
K - Capital: capital available in period t is k_{t-1}
a - Accumulated labor: value in period t is a_{t-1}
*/


//----------------------------------------------------------------------
// Time-to-build length
//----------------------------------------------------------------------
@#define J = 4



//----------------------------------------------------------------------
// Variables
//----------------------------------------------------------------------
var

//-------------------------
// Core quantities
//-------------------------
y_h y_f // output
m_h m_f // value added in production block
z_h z_f // inventories
k_h k_f // capital
lambda_h lambda_f // technology
s_h s_f // new capital starts (time-to-build)
n_h n_f // labor
c_h c_f // consumption
h_h h_f // accumulated leisure (or leisure index)
a_h a_f // accumulated labor
x_h x_f // total investment spending (installments)
nx_h nx_f // net exports


//-------------------------
// Auxiliary objects
//-------------------------
MPK_h MPK_f
MPZ_h MPZ_f
MPL_h MPL_f
Uc_h Uc_f
Uh_h Uh_f

//-------------------------
// Multipliers
//-------------------------
Lambda // global resource multiplier
;



//----------------------------------------------------------------------
// Exogenous shocks
//----------------------------------------------------------------------
varexo E_H E_F;



//----------------------------------------------------------------------
// Parameters
//----------------------------------------------------------------------
parameters

beta_h beta_f // discount factor
gamma_h gamma_f // CRRA / curvature (as coded)
mu_h mu_f // consumption-leisure share
alpha_h alpha_f // weight in leisure accumulation
eta_h eta_f // decay in labor/leisure accumulation
omega_h omega_f // country weight (can impose omega_f=1-omega_h if desired)
theta_h theta_f // capital share in value-added
sigma_h sigma_f // inventory term coefficient
nu_h nu_f // CES parameter (inventories vs value added)
delta_h delta_f // depreciation
phi_h phi_f // installment shares (time-to-build)


rho_H_H
rho_H_F
rho_F_F
rho_F_H
;


//----------------------------------------------------------------------
// Parameter values
//----------------------------------------------------------------------
beta_h = 0.99;
mu_h = 0.34;        // In time-use data, working time share is about 30%
gamma_h = -1.0;     // Risk aversion / IES-related curvature (Hansen 1988 suggests roughly -2 to 0.5 in this coding)
alpha_h = 1;
theta_h = 0.36;     // Labor share
nu_h = 3;           // Elasticity parameter between inventories and value added (in the CES aggregator)
sigma_h = 0.01;
delta_h = 0.025;    // Implies capital-output ratio around 10 (rough calibration target)
eta_h = 0.5;

phi_h = 1/@{J};
omega_h = 0.5;

beta_f = 0.99;
mu_f = 0.34;
gamma_f = -1.0;
alpha_f = 1;
eta_f = 0.5;
theta_f = 0.36;
nu_f = 3;
sigma_f = 0.01;
delta_f = 0.025;
phi_f = 1/@{J};
omega_f = 0.5;

// Technology shock coefficients calibrated from US and Europe business cycle statistics
rho_H_H = 0.906;
rho_F_F = 0.906;
rho_H_F = 0.088;
rho_F_H = 0.088;





//----------------------------------------------------------------------
// Model
//----------------------------------------------------------------------
model;

//====================================================================
// Home block
//====================================================================

//--- Production: value added + final good aggregator ---
m_h = lambda_h * k_h(-1)^theta_h * n_h^(1-theta_h);
y_h = (m_h^(-nu_h) + sigma_h*z_h(-1)^(-nu_h))^(-1/nu_h);

//--- Capital accumulation with time-to-build ---
k_h = (1-delta_h)*k_h(-1) + s_h(-@{J}+1);

//--- Installment investment spending ---
x_h =
@# for lag in (-J+1):0
+ phi_h * s_h(@{lag})
@# endfor
;

//--- National accounting identity ---
nx_h = y_h - (c_h + x_h + z_h - z_h(-1));



//--- Leisure accumulation and labor accumulation recursion ---
h_h = 1 - alpha_h*n_h - (1-alpha_h)*eta_h*a_h;
a_h = (1-eta_h) * a_h(-1) + n_h;

//--- Auxiliary objects (for compact FOCs) ---
MPZ_h = y_h^(nu_h+1) * sigma_h * z_h(-1)^(-nu_h-1);
MPK_h = y_h^(nu_h+1) * m_h^(-nu_h) * theta_h / k_h(-1);
MPL_h = y_h^(nu_h+1) * m_h^(-nu_h) * (1-theta_h) / n_h;

Uc_h = (c_h^mu_h * h_h^(1-mu_h))^(gamma_h-1) * mu_h * c_h^(mu_h-1) * h_h^(1-mu_h);
Uh_h = (c_h^mu_h * h_h^(1-mu_h))^(gamma_h-1) * (1-mu_h) * h_h^(-mu_h) * c_h^(mu_h);


//--- FOCs / equilibrium conditions (Home) ---
Lambda = omega_h * Uc_h; // consumption
Lambda = beta_h * Lambda(+1) * (MPZ_h(+1) + 1); // inventories
(alpha_h - eta_h)*beta_h*omega_h*Uh_h(+1) - alpha_h*omega_h*Uh_h = beta_h*(1-eta_h)*Lambda(+1)*MPL_h(+1) - Lambda*MPL_h;



0 = // capital start (time-to-build)
@# for lag in 0:(J-1)
+ beta_h^@{lag} * Lambda(+@{lag}) * phi_h
@# endfor
@# for lag in 1:J
- beta_h^@{lag} * Lambda(+@{lag}) * phi_h * (1-delta_h)
@# endfor
- beta_h^@{J} * Lambda(+@{J}) * MPK_h(+@{J})
;



//====================================================================
// Foreign block (mirror of Home): replace _h -> _f
//====================================================================

//--- Production: value added + final good aggregator ---
m_f = lambda_f * k_f(-1)^theta_f * n_f^(1-theta_f);
y_f = (m_f^(-nu_f) + sigma_f*z_f(-1)^(-nu_f))^(-1/nu_f);

//--- Capital accumulation with time-to-build ---
k_f = (1-delta_f)*k_f(-1) + s_f(-@{J}+1);

//--- Installment investment spending ---
x_f =
@# for lag in (-J+1):0
+ phi_f * s_f(@{lag})
@# endfor
;

//--- National accounting identity ---
nx_f = y_f - (c_f + x_f + z_f - z_f(-1));



//--- Leisure accumulation and labor accumulation recursion ---
h_f = 1 - alpha_f*n_f - (1-alpha_f)*eta_f*a_f;
a_f = (1-eta_f) * a_f(-1) + n_f;

//--- Auxiliary objects (for compact FOCs) ---
MPZ_f = y_f^(nu_f+1) * sigma_f * z_f(-1)^(-nu_f-1);
MPK_f = y_f^(nu_f+1) * m_f^(-nu_f) * theta_f / k_f(-1);
MPL_f = y_f^(nu_f+1) * m_f^(-nu_f) * (1-theta_f) / n_f;

Uc_f = (c_f^mu_f * h_f^(1-mu_f))^(gamma_f-1) * mu_f * c_f^(mu_f-1) * h_f^(1-mu_f);
Uh_f = (c_f^mu_f * h_f^(1-mu_f))^(gamma_f-1) * (1-mu_f) * h_f^(-mu_f) * c_f^(mu_f);


//--- FOCs / equilibrium conditions (Foreign) ---
Lambda = omega_f * Uc_f; // consumption
Lambda = beta_f * Lambda(+1) * (MPZ_f(+1) + 1); // inventories
(alpha_f - eta_f)*beta_f*omega_f*Uh_f(+1) - alpha_f*omega_f*Uh_f
    = beta_f*(1-eta_f)*Lambda(+1)*MPL_f(+1) - Lambda*MPL_f;



0 = // capital start (time-to-build)
@# for lag in 0:(J-1)
+ beta_f^@{lag} * Lambda(+@{lag}) * phi_f
@# endfor
@# for lag in 1:J
- beta_f^@{lag} * Lambda(+@{lag}) * phi_f * (1-delta_f)
@# endfor
- beta_f^@{J} * Lambda(+@{J}) * MPK_f(+@{J})
;







//====================================================================
// (4C) Global market clearing + shocks
//====================================================================

//--- Global resource constraint ---
y_h + y_f = c_h + c_f + x_h + x_f + (z_h - z_h(-1)) + (z_f - z_f(-1));

//--- Technology shock processes ---
(lambda_h - 1) = rho_H_H*(lambda_h(-1) - 1) + rho_H_F*(lambda_f(-1) - 1) + E_H;
(lambda_f - 1) = rho_F_F*(lambda_f(-1) - 1) + rho_F_H*(lambda_h(-1) - 1) + E_F;

end;







//----------------------------------------------------------------------
// Initial values for steady state
//----------------------------------------------------------------------
initval;

//=========================
// Home
//=========================
lambda_h = 1;
k_h = 1;
n_h = 0.5;
z_h = 1;

// When alpha=1, 'a' does not affect 'h', but we still set a steady-consistent value: a = n/eta
a_h = n_h/eta_h;

// Production block (consistent)
m_h = lambda_h * k_h^theta_h * n_h^(1-theta_h);
y_h = (m_h^(-nu_h) + sigma_h*z_h^(-nu_h))^(-1/nu_h);

// Time-to-build steady consistency: k' = k  => s = delta*k
s_h = delta_h * k_h;

// Installment investment: x = sum_{j=0}^{J-1} phi*s = J*phi*s (steady)
x_h = @{J} * phi_h * s_h;

// Leisure (works for general alpha; with alpha=1 this implies h=1-n)
h_h = 1 - alpha_h*n_h - (1-alpha_h)*eta_h*a_h;

// Net exports and consumption at steady: Delta z = 0 and set nx=0 => c = y - x
nx_h = 0;
c_h = y_h - x_h;



// Derived variables (must be in var and defined in model as equations)
MPZ_h = y_h^(nu_h+1) * sigma_h * z_h^(-nu_h-1);
MPK_h = y_h^(nu_h+1) * m_h^(-nu_h) * theta_h / k_h;
MPL_h = y_h^(nu_h+1) * m_h^(-nu_h) * (1-theta_h) / n_h;

Uc_h = (c_h^mu_h * h_h^(1-mu_h))^(gamma_h-1) * mu_h * c_h^(mu_h-1) * h_h^(1-mu_h);
Uh_h = (c_h^mu_h * h_h^(1-mu_h))^(gamma_h-1) * (1-mu_h) * h_h^(-mu_h) * c_h^(mu_h);



//=========================
// Foreign (symmetric)
//=========================
lambda_f = 1;
k_f = 1;
n_f = 0.5;
z_f = 1;

a_f = n_f/eta_f;

m_f = lambda_f * k_f^theta_f * n_f^(1-theta_f);
y_f = (m_f^(-nu_f) + sigma_f*z_f^(-nu_f))^(-1/nu_f);

s_f = delta_f * k_f;
x_f = @{J} * phi_f * s_f;

h_f = 1 - alpha_f*n_f - (1-alpha_f)*eta_f*a_f;

nx_f = 0;
c_f = y_f - x_f;



MPZ_f = y_f^(nu_f+1) * sigma_f * z_f^(-nu_f-1);
MPK_f = y_f^(nu_f+1) * m_f^(-nu_f) * theta_f / k_f;
MPL_f = y_f^(nu_f+1) * m_f^(-nu_f) * (1-theta_f) / n_f;

Uc_f = (c_f^mu_f * h_f^(1-mu_f))^(gamma_f-1) * mu_f * c_f^(mu_f-1) * h_f^(1-mu_f);
Uh_f = (c_f^mu_f * h_f^(1-mu_f))^(gamma_f-1) * (1-mu_f) * h_f^(-mu_f) * c_f^(mu_f);





// =========================
// Global multiplier + shocks
//=========================
Lambda = omega_h * Uc_h;
E_H = 0;
E_F = 0;

end;






//----------------------------------------------------------------------
// Steady state + checks
//----------------------------------------------------------------------
steady;
check;
model_info;








//----------------------------------------------------------------------
// Shocks / Simulation
//----------------------------------------------------------------------
shocks;
var E_H; stderr 0.00852;
var E_F; stderr 0.00852;
corr E_H, E_F = 0.258;
end;


// Save figures
//options_.nodisplay = 1;       // do not pop up figure windows
options_.graph_format = 'pdf';
options_.TeX = 0;

stoch_simul(order=1, irf=60) lambda_h y_h c_h x_h nx_h;

// Save simulation outputs
irfs = oo_.irfs;
steady_state = oo_.steady_state;
endo_names = M_.endo_names;
exo_names  = M_.exo_names;

save('dynare_irfs_v73.mat', 'irfs', 'steady_state', 'endo_names', 'exo_names', '-v7.3');
