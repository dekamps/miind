function izsh = izshikevich_backward( t, ws )
    % TODO : rename this function.
    
    g_nap = 0.25; %mS - %250.0; %nS
    g_na = 30;
    g_k = 1;
    theta_m = -47.1; %mV
    sig_m = -3.1; %mV
    theta_h = -59; %mV
    sig_h = 8; %mV
    tau_h = 1200; %ms 
    E_na = 55; %mV
    E_k = -80;
    C = 1; %uF - %1000; %pF
    g_l = 0.1; %mS - %100.0; %nS
    E_l = -64.0; %mV
    I = 0.0; %
    I_h = 0; %
    
    v = ws(1);
    h = ws(2);
    
    I_nap = -g_nap * h * (v - E_na) * ((1 + exp((-v-47.1)/3.1))^-1);
    I_l = -g_l*(v - E_l);
    I_na = -g_na * 0.7243 * (v - E_na) * (((1 + exp((v+35)/-7.8))^-1)^3);
    I_k = -g_k * (v - E_k) * (((1 + exp((v+28)/-15))^-1)^4);
    
    v_prime = ((I_nap + I_l + I_na + I_k) / C)+I;
    h_prime = (((1 + (exp((v - theta_h)/sig_h)))^(-1)) - h ) / (tau_h/cosh((v - theta_h)/(2*sig_h))) + I_h;
    
    izsh = [-v_prime; -h_prime];
end

