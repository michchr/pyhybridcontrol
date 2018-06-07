%% A
clear all;
close all;

%% B1

syms m_h C_w D_h T_w T_inf P_h U_h A_h T_h ts p1 p2

p1 = U_h * A_h;
p2 = m_h * C_w;

A = -(D_h * C_w + p1)/p2;
B = [C_w * T_w, 1, p1*T_inf]/p2;

% syms A;

Phi_k = exp(A*ts);
Gamma_k_3 = (A^(-1) * (exp(A*ts)-1) * B);
Gamma_k = Gamma_k_3(1:2);
H_k = Gamma_k_3(3);

%T_h(k+1) = Phi(k)*T_h(k) * T_k + Gamma(k)*[D_h(k), P_h(k)].' + H(k)

eval_Phi_k = matlabFunction(Phi_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h, D_h});
        
eval_Gamma_k = matlabFunction(Gamma_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h, T_w, D_h});
        
eval_H_k = matlabFunction(H_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h, T_w, T_inf, D_h});
        
%% B2

syms m_h C_w D_h T_w T_inf P_h U_h A_h T_h ts p1 p2

p1 = U_h * A_h;
p2 = m_h * C_w;

A2 = -p1/p2;
B2 = [-1, 1, p1*T_inf]/p2;

% syms A;

Phi2_k = simplify(exp(A2*ts));
Gamma2_k_3 = simplify(A2^(-1) * (exp(A2*ts)-1) * B2);
Gamma2_k = Gamma2_k_3(1:2);
H2_k = Gamma2_k_3(3);

%T_h(k+1) = Phi(k)*T_h(k) * T_k + Gamma(k)*[D_h(k), P_h(k)].' + H(k)

eval_Phi2_k = matlabFunction(Phi2_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h});
        
eval_Gamma2_k = matlabFunction(Gamma2_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h});
        
eval_H2_k = matlabFunction(H2_k,...
            'Vars', {ts, C_w, A_h, U_h, m_h, T_inf});
        
%% Test

ts = 60;
C_w = 4.1816 * 10^3;
A_h = 1;
U_h = 2.7;
m_h = 150;
D_h = 8/60;
T_w = 25;
T_inf = 25;
P_h = 0;

Phi_k = eval_Phi_k(ts, C_w, A_h, U_h, m_h, D_h);
Gamma_k = eval_Gamma_k(ts, C_w, A_h, U_h, m_h, T_w, D_h);
H_k = eval_H_k(ts, C_w, A_h, U_h, m_h, T_w, T_inf, D_h);

Phi2_k = eval_Phi2_k(ts, C_w, A_h, U_h, m_h);
Gamma2_k = eval_Gamma2_k(ts, C_w, A_h, U_h, m_h);
H2_k = eval_H2_k(ts, C_w, A_h, U_h, m_h, T_inf);

tf = 50;
T0 = 65;

Th1 = zeros(1,tf);
Th2 = zeros(1,tf);
Qdem1 = zeros(1,tf);
Qdem2 = zeros(1,tf);
Th1(1) = T0;
Th2(1) = T0;
Qdem1(1) = D_h*C_w*(Th1(1)-T_w);
Qdem2(1) = D_h*C_w*(Th2(1)-T_w);
D_h_list = zeros(1,tf);
Heat_loss = zeros(1,tf);
Heat_loss(1) = 0;
for k = 1:(tf-1)
    if Th1(k) < 50
        P_h = 3000;
    else 
        P_h = 0;
    end
    Th1(k+1) = Phi_k*Th1(k) + Gamma_k*[D_h;P_h] + H_k;
    Th2(k+1) = Phi2_k*Th2(k) + Gamma2_k*[Qdem2(k);P_h] + H2_k;
    Heat_loss(k) = (Th1(k+1) - Phi2_k*Th1(k) - H2_k - Gamma2_k(2)*P_h)/(Gamma2_k(1));
    Qdem1(k+1) = D_h*C_w*(Th1(k+1)-T_w);
    Qdem2(k+1) = D_h*C_w*(Th2(k+1)-T_w);
    
    D_h_list(k) = Heat_loss(k)/(C_w*(Th1(k)-T_w));
end

plot(Th1,'.')
hold on
plot(Th2,'*')

D_h_list

% [Th1', Qdem1',zeros(tf,1), Qdem2', Th2']