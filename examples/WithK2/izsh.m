%#ok<*NOPTS>
% Reset should be -50, threshold is -40,  d = 0 (for now)
timestep = 2;
eff = 0.1;
effInhib = -0.1;
effBig = -17.0;
jmpId = fopen('rinzel_0.1.jmp', 'w');
fprintf(jmpId, '<Jump>\n');
fprintf(jmpId, '<Efficacy>%f,0</Efficacy>\n', eff);
fprintf(jmpId, '<Transitions>\n');
fprintf(jmpId, '%i,%i %f 0\n', 0, 0, 0.1);
    
jmpInhibId = fopen('rinzel_-0.1.jmp', 'w');
fprintf(jmpInhibId, '<Jump>\n');
fprintf(jmpInhibId, '<Efficacy>%f,0</Efficacy>\n', effInhib);
fprintf(jmpInhibId, '<Transitions>\n');
fprintf(jmpInhibId, '%i,%i %f 0\n', 0, 0, -0.1);

jmpBigId = fopen('rinzel_-17.0.jmp', 'w');
fprintf(jmpBigId, '<Jump>\n');
fprintf(jmpBigId, '<Efficacy>%f,0</Efficacy>\n', effBig);
fprintf(jmpBigId, '<Transitions>\n');
fprintf(jmpBigId, '%i,%i %f 0\n', 0, 0, -17.0);

revId = fopen('rinzel.rev', 'w');
fprintf(revId, '<Mapping Type="Reversal">\n');

outId = fopen('rinzel.mesh', 'w');
fprintf(outId, 'ignore\n');
fprintf(outId, '%f\n', timestep/1000);

formatSpec = '%.12f ';
strip = 1;
cell = 0;

g_nap = 0.25; %mS - %250.0; %nS
theta_m = -47.1; %mV

sig_m = -3.1; %mV
theta_h = -59; %mV
sig_h = 8; %mV
tau_h = 1200; %ms 
E_na = 55; %mV
C = 1; %uF - %1000; %pF
% Depending on how much noise we expect, E_l should be reduced to the point
% where there is still no activity - that is, the stationary point comes before the v
% nullcline peak. (higher I => lower v nullcline peak - stationary point moves to the right, 
% lower E_l => higher nullcline peak - stationary point moves to the left)
% ***Note for each -1 in E_l, I must increase by g_l to keep the same stationary point.*** 
g_l = 0.1; %mS
E_l = -64.0; %mV
I = 0.0; %
g_na = 30;
theta_m_na = -35;
sig_m_na = -7.8;
E_k= -80;
g_k = 1;


stat_a = 0;
stat_b = 0;
stat_c = 0;
stat_d = 0;

centre_point_x = -52.85;

centre_point_y_hi = ((g_l.*(centre_point_x-E_l)) + (g_k.*(((1./(1 + exp((centre_point_x+28)/-15)))).^4).*(centre_point_x-E_k)) + (g_na.*0.7243.*(((1./(1 + exp((centre_point_x-theta_m_na)/sig_m_na)))).^3).*(centre_point_x-E_na)) - I ) ./ ((centre_point_x-E_na).*-g_nap.*(1./(1 + exp((centre_point_x-theta_m)/sig_m)))) + 0.001;
centre_point_y_lo = ((g_l.*(centre_point_x-E_l)) + (g_k.*(((1./(1 + exp((centre_point_x+28)/-15)))).^4).*(centre_point_x-E_k)) + (g_na.*0.7243.*(((1./(1 + exp((centre_point_x-theta_m_na)/sig_m_na)))).^3).*(centre_point_x-E_na)) - I ) ./ ((centre_point_x-E_na).*-g_nap.*(1./(1 + exp((centre_point_x-theta_m)/sig_m)))) - 0.001;

start_point_x = -64.2;

start_point_y = ((g_l.*(start_point_x-E_l)) + (g_k.*(((1./(1 + exp((start_point_x+28)/-15)))).^4).*(start_point_x-E_k)) + (g_na.*0.7243.*(((1./(1 + exp((start_point_x-theta_m_na)/sig_m_na)))).^3).*(start_point_x-E_na)) - I ) ./ ((start_point_x-E_na).*-g_nap.*(1./(1 + exp((start_point_x-theta_m)/sig_m))));

full_time = 7000;

strip_cell_strip_width = 50;

tspan = 0:timestep:full_time;
[t1,s1] = ode23s(@izshikevich, tspan, [start_point_x start_point_y]);

x1 = s1(:,1);
y1 = s1(:,2);

time_end = full_time;

% if find(x1>centre_point_x,1,'first') > 1
%     time_end = find(x1>centre_point_x,1,'first')*timestep
% end

% Nullcline strip (lower)
right_curve = [];
left_curve = [];

svs_1 = [];
sus_1 = [];
svs_2 = [];
sus_2 = [];

nullcline_strip_length = (time_end/timestep)-100;
nullcline_strip_length = nullcline_strip_length+6;

for i = 2:nullcline_strip_length
    fwd_vec = s1(i+1,:)-s1(i,:);
    right_vec = [fwd_vec(2),-fwd_vec(1)];
    right_vec =  0.01 * right_vec / norm(right_vec);
    
    left_vec = -right_vec * 0.5;
    
    p_left = s1(i,:) + [-0.01,0.0];
    p_right = s1(i,:) + [0.01,0.0];
    
    right_curve = [right_curve;p_right];
    left_curve = [left_curve;p_left];
    
    a = p_left(1);
    b = p_left(2);
    c = p_right(1);
    d = p_right(2);
    
    svs_1 = [svs_1, a];
    sus_1 = [sus_1, b];
    
    svs_2 = [svs_2, c];
    sus_2 = [sus_2, d];
    
    if i > 2
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
        hold on;
end
% 
stat_a = p_right;
stat_d = p_left;

plot(right_curve(:,1),right_curve(:,2),'k');
plot(left_curve(:,1),left_curve(:,2),'k');

fprintf(outId, formatSpec, svs_1);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_1);
fprintf(outId, '\n');

fprintf(outId, formatSpec, svs_2);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_2);
fprintf(outId, '\n');

fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 0,0, 1.0);

strip = strip + 1;
cell = 0;

fprintf(outId, 'closed\n');

% (LEFT SECTION) Starting from Nullcline strip (left) going backwards to the left
% 
v0 = left_curve(5+(1*strip_cell_strip_width),1);
h0 = left_curve(5+(1*strip_cell_strip_width),2);
[t1,prev_s2] = ode23s(@izshikevich_backward, tspan, [v0 h0]);

for i = 5+(1*strip_cell_strip_width):strip_cell_strip_width:nullcline_strip_length-(1*strip_cell_strip_width)
    v1 = left_curve(i+strip_cell_strip_width,1);
    h1 = left_curve(i+strip_cell_strip_width,2);
    tspan = 0:timestep:500;

    s1 = prev_s2;
    [t2,s2] = ode23s(@izshikevich_backward, tspan, [v1 h1]);
    prev_s2 = s2;
    
    x1 = s1(:,1);
    
    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end


    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end

        plot([a c],[b d],'k');
        hold on;
    end
    
    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');
    
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 1,i-1, 1.0/strip_cell_strip_width);
    for j = 1:strip_cell_strip_width-2
        fprintf(revId, '%i,%i\t%i,%i\t%f\n',strip, 0, 1,i+j-1, 1.0/strip_cell_strip_width);
    end
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0,1,i+strip_cell_strip_width-1-1, 1.0/strip_cell_strip_width);

    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    plot(x2,y2,'k');
    hold on;
end

%(LOWER SECTION) Starting from the nullcline strip (right) and going
%backwards to the right.

small_strip_width = strip_cell_strip_width;
small_strip_start = 15;

v0 = right_curve(5+(small_strip_width),1);
h0 = right_curve(5+(small_strip_width),2);
[t1,prev_s2] = ode23s(@izshikevich_backward, tspan, [v0 h0]);

for i = 5+(small_strip_width):small_strip_width:((small_strip_start * strip_cell_strip_width)-small_strip_width)+5
    v1 = right_curve(i+small_strip_width,1);
    h1 = right_curve(i+small_strip_width,2);
    tspan = 0:timestep:400; %170 %- (30 * (max(0,(((v1-right_curve(5,1)) / (right_curve((small_strip_start * strip_cell_strip_width)-small_strip_width,1) - right_curve(5,1)))))^2));
    
    s1 = prev_s2;
    [t2,s2] = ode23s(@izshikevich_backward, tspan, [v1 h1]);
    prev_s2 = s2;

    x1 = s1(:,1);
    
    cut = find(x1>10,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2>10,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');

    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 1,i-1, 1.0/small_strip_width);
    for j = 1:small_strip_width-2
        fprintf(revId, '%i,%i\t%i,%i\t%f\n',strip, 0, 1,i+j-1, 1.0/small_strip_width);
    end
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0,1,i+small_strip_width-1-1, 1.0/small_strip_width);

    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    plot(x2,y2,'k');
    hold on;
end

for i = 5+(small_strip_start*strip_cell_strip_width):strip_cell_strip_width:5+nullcline_strip_length-strip_cell_strip_width
    v1 = right_curve(i+strip_cell_strip_width,1);
    h1 = right_curve(i+strip_cell_strip_width,2);
    tspan = 0:timestep:400; %160 %+ (50 * (max(0,(((v1-right_curve(((small_strip_start * strip_cell_strip_width)-small_strip_width),1)) / (right_curve((nullcline_strip_length-strip_cell_strip_width),1)-right_curve(((small_strip_start * strip_cell_strip_width)-small_strip_width),1)))))^2));

    s1 = prev_s2;
    [t2,s2] = ode23s(@izshikevich_backward, tspan, [v1 h1]);
    prev_s2 = s2;

    x1 = s1(:,1);
    
    cut = find(x1>10,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2>10,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end

        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');

    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 1,i-1, 1.0/strip_cell_strip_width);
    for j = 1:strip_cell_strip_width-2
        fprintf(revId, '%i,%i\t%i,%i\t%f\n',strip, 0, 1,i+j-1, 1.0/strip_cell_strip_width);
    end
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0,1,i+strip_cell_strip_width-1-1, 1.0/strip_cell_strip_width);

    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    plot(x2,y2,'k');
    hold on;
end

% % Nullcline strip (upper)
% 
start_point_x = -61.52;

start_point_y = ((g_l.*(start_point_x-E_l)) + (g_k.*(((1./(1 + exp((start_point_x+28)/-15)))).^4).*(start_point_x-E_k)) + (g_na.*0.7243.*(((1./(1 + exp((start_point_x-theta_m_na)/sig_m_na)))).^3).*(start_point_x-E_na)) - I ) ./ ((start_point_x-E_na).*-g_nap.*(1./(1 + exp((start_point_x-theta_m)/sig_m))));

full_time = 3920;

tspan = 80:timestep:full_time;
[t1,s1] = ode23s(@izshikevich, tspan, [start_point_x start_point_y]);

x1 = s1(:,1);
y1 = s1(:,2);

time_end = full_time;

right_curve = [];
left_curve = [];

svs_1 = [];
sus_1 = [];
svs_2 = [];
sus_2 = [];

nullcline_strip_length = (time_end/timestep)-100;
nullcline_strip_length = nullcline_strip_length + 6;

nullcline_strip_num = strip;

for i = 2:nullcline_strip_length
    fwd_vec = s1(i+1,:)-s1(i,:);
    right_vec = [fwd_vec(2),-fwd_vec(1)];
    right_vec =  0.01 * right_vec / norm(right_vec);
    
    left_vec = -right_vec;
    
%     p_left = s1(i,:) + [0.005 + ((1-(i/nullcline_strip_length))*0.05),0.0];
%     p_right = s1(i,:) + [-0.005 - ((1-(i/nullcline_strip_length))*0.05),0.0];
    
    p_left = s1(i,:) + [0.01,0.0];
    p_right = s1(i,:) + [-0.01,0.0];
    
    right_curve = [right_curve;p_right];
    left_curve = [left_curve;p_left];
    
    a = p_left(1);
    b = p_left(2);
    c = p_right(1);
    d = p_right(2);
    
    svs_1 = [svs_1, a];
    sus_1 = [sus_1, b];
    
    svs_2 = [svs_2, c];
    sus_2 = [sus_2, d];
    
    if i > 2
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
        hold on;
end

stat_b = p_left;
stat_c = p_right;

plot(right_curve(:,1),right_curve(:,2),'k');
plot(left_curve(:,1),left_curve(:,2),'k');

fprintf(outId, formatSpec, svs_1);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_1);
fprintf(outId, '\n');

fprintf(outId, formatSpec, svs_2);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_2);
fprintf(outId, '\n');

fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 0,0, 1.0);

strip = strip + 1;
cell = 0;

fprintf(outId, 'closed\n');

%(LEFT SECTION) Starting from Nullcline strip (upper) going backwards to the left
v0 = right_curve(32,1);
h0 = right_curve(32,2);
[t1,prev_s2] = ode45(@izshikevich_backward, tspan, [v0 h0]);
small_width = strip_cell_strip_width;

left_last_row_1 = prev_s2;

for i = 32:small_width:nullcline_strip_length-small_width
    v1 = right_curve(i+small_width,1);
    h1 = right_curve(i+small_width,2);
    tspan = 0:timestep:500;

    s1 = prev_s2;
    [t2,s2] = ode45(@izshikevich_backward, tspan, [v1 h1]);
    prev_s2 = s2;

    x1 = s1(:,1);
    
    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end
    
    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');
    
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, nullcline_strip_num,i-1, 1.0/small_width);
    for j = 1:small_width-2
        fprintf(revId, '%i,%i\t%i,%i\t%f\n',strip, 0, nullcline_strip_num,i+j-1, 1.0/small_width);
    end
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0,nullcline_strip_num,i+small_width-1-1, 1.0/small_width);
   
    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    plot(x2,y2,'k');
    hold on;
end

v0 = left_curve(58,1);
h0 = left_curve(58,2);
tspan = 0:timestep:500;
[t1,prev_s2] = ode45(@izshikevich_backward, tspan, [v0 h0]);

for i = 58:strip_cell_strip_width:nullcline_strip_length-strip_cell_strip_width
    v1 = left_curve(i+strip_cell_strip_width,1);
    h1 = left_curve(i+strip_cell_strip_width,2);
    tspan = 0:timestep:500;

    s1 = prev_s2;
    [t2,s2] = ode45(@izshikevich_backward, tspan, [v1 h1]);
    prev_s2 = s2;

    x1 = s1(:,1);
    y1 = s1(:,2);
    
    cut = find(x1 < -90 | y1 < 0 ,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    y2 = s2(:,2);
    
    cut = find(x2 < -90 | y2 < 0,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end

        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');

    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, nullcline_strip_num,i-1, 1.0/strip_cell_strip_width);
    for j = 1:strip_cell_strip_width-2
        fprintf(revId, '%i,%i\t%i,%i\t%f\n',strip, 0, nullcline_strip_num,i+j-1, 1.0/strip_cell_strip_width);
    end
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0,nullcline_strip_num,i+strip_cell_strip_width-1-1, 1.0/strip_cell_strip_width);

    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    plot(x2,y2,'k');
    hold on;
end

% Backwards Snake

vstep = 0.25;
v_init = -61.548;
h_init = 0.5965;

strip_count = 0;
for h0 = h_init:0.002:0.626
    cell_limit = 900;
    
    v0 = v_init;
    tspan = 0:timestep:cell_limit;
    
    [t1,s1] = ode45(@izshikevich_backward, tspan, [v0 h0]);
    [t2,s2] = ode45(@izshikevich_backward, tspan, [v0+0.0201 h0+0.002]);
    
    s2_first = s2;

    x1 = s1(:,1);
    
    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);
    
    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);
    
    plot(x1,y1,'k');
    hold on;
    
    cell_limit = 150 + (log((((h0-h_init) / (0.625-h_init))*10)+1)/4 * 1100);
    
    v0 = v_init;
    v_init = v_init + 0.0201;
    tspan = 0:timestep:cell_limit;
    
    [t1,s1] = ode23s(@izshikevich, tspan, [v0 h0]);
    [t2,s2] = ode23s(@izshikevich, tspan, [v0+0.0201 h0+0.002]);

    x1 = s1(:,1);
    
    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);
    
    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

%     svs_1 = [];
%     sus_1 = [];
%     svs_2 = [];
%     sus_2 = [];
    for t = 2:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];
        
        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');
    
    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');
    
    fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, nullcline_strip_num,20, 1.0);
    
    strip = strip + 1;
    cell = 0;
    
    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);
    

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    hold on;
end

snake_top_right = s2;
snake_top_left = s2_first;

% Stationary Cell strips

v0 = stat_a(1);
h0 = stat_a(2);
v1 = stat_b(1);
h1 = stat_b(2);
tspan = 0:timestep:350;
 
[t1,s1] = ode23s(@izshikevich_backward, tspan, [v0 h0]);
[t2,s2] = ode23s(@izshikevich_backward, tspan, [v1 h1]);
 
x1 = s1(:,1);
 
cut = find(x1<-120,1,'first');
if cut > 1
    x1 = x1(1:cut-1);
    y1 = s1(1:cut-1,2);
else
    x1 = s1(:,1);
    y1 = s1(:,2);
end
 
x2 = s2(:,1);
 
cut = find(x2<-120,1,'first');
if cut > 1
    x2 = x2(1:cut-1);
    y2 = s2(1:cut-1,2);
else
    x2 = s2(:,1);
    y2 = s2(:,2);
end
 
svs_1 = [];
sus_1 = [];
svs_2 = [];
sus_2 = [];
for t = 1:min(length(x2),length(x1))
    a = x1(t);
    b = y1(t);
    c = x2(t);
    d = y2(t);
 
    svs_1 = [svs_1, x1(t)];
    sus_1 = [sus_1, y1(t)];
 
    svs_2 = [svs_2, x2(t)];
    sus_2 = [sus_2, y2(t)];
    
    if t > 1
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
    hold on;
end
 
svs_1 = fliplr(svs_1);
sus_1 = fliplr(sus_1);
 
svs_2 = fliplr(svs_2);
sus_2 = fliplr(sus_2);
 
fprintf(outId, formatSpec, svs_1);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_1);
fprintf(outId, '\n');
 
fprintf(outId, formatSpec, svs_2);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_2);
fprintf(outId, '\n');
fprintf(outId, 'closed\n');
 
fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 0,0, 1);
 
strip = strip + 1;
cell = 0;

axis fill
xlim([-70 -40]);
ylim([0 1.2]);
 
title('Rinzel');
ylabel('h');
xlabel('v');
 
plot(x1,y1,'k');
plot(x2,y2,'k');
hold on;
 
v0 = stat_c(1);
h0 = stat_c(2);
v1 = stat_d(1);
h1 = stat_d(2);
tspan = 0:timestep:800;
 
[t1,s1] = ode23s(@izshikevich_backward, tspan, [v0 h0]);
[t2,s2] = ode23s(@izshikevich_backward, tspan, [v1 h1]);
 
x1 = s1(:,1);
 
cut = find(x1<-120,1,'first');
if cut > 1
    x1 = x1(1:cut-1);
    y1 = s1(1:cut-1,2);
else
    x1 = s1(:,1);
    y1 = s1(:,2);
end
 
x2 = s2(:,1);
 
cut = find(x2<-120,1,'first');
if cut > 1
    x2 = x2(1:cut-1);
    y2 = s2(1:cut-1,2);
else
    x2 = s2(:,1);
    y2 = s2(:,2);
end
 
svs_1 = [];
sus_1 = [];
svs_2 = [];
sus_2 = [];
for t = 1:min(length(x2),length(x1))
    a = x1(t);
    b = y1(t);
    c = x2(t);
    d = y2(t);
 
    svs_1 = [svs_1, x1(t)];
    sus_1 = [sus_1, y1(t)];
 
    svs_2 = [svs_2, x2(t)];
    sus_2 = [sus_2, y2(t)];
    
    if t > 1
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);
        
        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
    hold on;
end
 
svs_1 = fliplr(svs_1);
sus_1 = fliplr(sus_1);
 
svs_2 = fliplr(svs_2);
sus_2 = fliplr(sus_2);
 
fprintf(outId, formatSpec, svs_1);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_1);
fprintf(outId, '\n');
 
fprintf(outId, formatSpec, svs_2);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_2);
fprintf(outId, '\n');
fprintf(outId, 'closed\n');
 
fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 0,0, 1);
strip = strip + 1;
cell = 0;

axis fill
xlim([-70 -40]);
ylim([0 1.2]);
 
title('Rinzel');
ylabel('h');
xlabel('v');
 
plot(x1,y1,'k');
plot(x2,y2,'k');
hold on;

% Right upper

v_init = -20.0;
h_init = 0.62;

%joiner
[t2,upper_to_snake_top] = ode23s(@izshikevich_backward, tspan, [v_init h_init]);

x1 = snake_top_left(:,1);
cut = find(x1<-120,1,'first');
if cut > 1
    x1 = x1(1:cut-1);
    y1 = snake_top_left(1:cut-1,2);
else
    x1 = snake_top_left(:,1);
    y1 = snake_top_left(:,2);
end

x2 = upper_to_snake_top(:,1);
cut = find(x2<-120,1,'first');
if cut > 1
    x2 = x2(1:cut-1);
    y2 = upper_to_snake_top(1:cut-1,2);
else
    x2 = upper_to_snake_top(:,1);
    y2 = upper_to_snake_top(:,2);
end

x3 = snake_top_right(:,1);
cut = find(x3<-120,1,'first');
if cut > 1
    x3 = x3(1:cut-1);
    y3 = snake_top_right(1:cut-1,2);
else
    x3 = snake_top_right(:,1);
    y3 = snake_top_right(:,2);
end

x1 = flipud(x1);
y1 = flipud(y1);
x2 = flipud(x2);
y2 = flipud(y2);

svs_1 = [];
sus_1 = [];
svs_2 = [];
sus_2 = [];
for t = 1:min(length(x2),length(x1))
    a = x1(t);
    b = y1(t);
    c = x2(t);
    d = y2(t);

    svs_1 = [svs_1, x1(t)];
    sus_1 = [sus_1, y1(t)];

    svs_2 = [svs_2, x2(t)];
    sus_2 = [sus_2, y2(t)];
    
    if t > 0
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%f,%f %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%f,%f %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
    hold on;
end

for t = 1:75
    a = x3(t+1);
    b = y3(t+1);
    c = x2(t+length(x1));
    d = y2(t+length(x1));

    svs_1 = [svs_1, x3(t+1)];
    sus_1 = [sus_1, y3(t+1)];

    svs_2 = [svs_2, x2(t+length(x1))];
    sus_2 = [sus_2, y2(t+length(x1))];
    
    if t > 1
        jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
        fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

        jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
        
        jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
        fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

        cell = cell + 1;
    end
    
    plot([a c],[b d],'k');
    hold on;
end

fprintf(outId, formatSpec, svs_1);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_1);
fprintf(outId, '\n');

fprintf(outId, formatSpec, svs_2);
fprintf(outId, '\n');
fprintf(outId, formatSpec, sus_2);
fprintf(outId, '\n');
fprintf(outId, 'closed\n');

fprintf(revId, '%i,%i\t%i,%i\t%f\n', strip, 0, 127,31, 1.0);

strip = strip + 1;
cell = 0;

strip_count = 0;
for h0 = h_init:0.02:0.7

    strip_count = strip_count + 1;

    cell_limit = 5000;

    v0 = v_init;
    tspan = 0:timestep:cell_limit;

    [t1,s1] = ode23s(@izshikevich_backward, tspan, [v0 h0]);
    [t2,s2] = ode23s(@izshikevich_backward, tspan, [v0 h0 + 0.02]);

    x1 = s1(:,1);

    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);

    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];

        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);

    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');

    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');

    strip = strip + 1;
    cell = 0;

    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);


    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    hold on;
end

%right lower

v_init = -20.0;
h_init = 0.6199;

strip_count = 0;
for h0 = h_init:-0.02:-0.3

    strip_count = strip_count + 1;

    cell_limit = 230;

    v0 = v_init;
    tspan = 0:timestep:cell_limit;

    [t1,s1] = ode23s(@izshikevich_backward, tspan, [v0 h0]);
    [t2,s2] = ode23s(@izshikevich_backward, tspan, [v0 h0 - 0.02]);

    x1 = s1(:,1);

    cut = find(x1<-120,1,'first');
    if cut > 1
        x1 = x1(1:cut-1);
        y1 = s1(1:cut-1,2);
    else
        x1 = s1(:,1);
        y1 = s1(:,2);
    end

    x2 = s2(:,1);

    cut = find(x2<-120,1,'first');
    if cut > 1
        x2 = x2(1:cut-1);
        y2 = s2(1:cut-1,2);
    else
        x2 = s2(:,1);
        y2 = s2(:,2);
    end

    svs_1 = [];
    sus_1 = [];
    svs_2 = [];
    sus_2 = [];
    for t = 1:min(length(x2),length(x1))
        a = x1(t);
        b = y1(t);
        c = x2(t);
        d = y2(t);

        svs_1 = [svs_1, x1(t)];
        sus_1 = [sus_1, y1(t)];

        svs_2 = [svs_2, x2(t)];
        sus_2 = [sus_2, y2(t)];
        
        if t > 1
            jmp = -0.05 * eff * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 10);
            fprintf(jmpId, '%i,%i %f 0\n', strip, cell, jmp);

            jmpInhib = 0.05 * effInhib * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpInhibId, '%i,%i %f 0\n', strip, cell, jmpInhib);
            
            jmpBig = 0.05 * effBig * (((((a+b)/2.0) + ((c+d)/2.0)) / 2.0) + 70);
            fprintf(jmpBigId, '%i,%i %f 0\n', strip, cell, jmpBig);

            cell = cell + 1;
        end
        
        plot([a c],[b d],'k');
        hold on;
    end

    svs_1 = fliplr(svs_1);
    sus_1 = fliplr(sus_1);

    svs_2 = fliplr(svs_2);
    sus_2 = fliplr(sus_2);

    fprintf(outId, formatSpec, svs_1);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_1);
    fprintf(outId, '\n');

    fprintf(outId, formatSpec, svs_2);
    fprintf(outId, '\n');
    fprintf(outId, formatSpec, sus_2);
    fprintf(outId, '\n');
    fprintf(outId, 'closed\n');

    strip = strip + 1;
    cell = 0;

    axis fill
    xlim([-70 -40]);
    ylim([0 1.2]);

    title('Rinzel');
    ylabel('h');
    xlabel('v');

    plot(x1,y1,'k');
    hold on;
end
 
fprintf(outId, 'end\n');

%nullclines
vv = -120:0.01:0;
    
v_nulls = ((g_l.*(vv-E_l)) + (g_k.*(((1./(1 + exp((vv+28)/-15)))).^4).*(vv-E_k)) + (g_na.*0.7243.*(((1./(1 + exp((vv-theta_m_na)/sig_m_na)))).^3).*(vv-E_na)) - I ) ./ ((vv-E_na).*-g_nap.*(1./(1 + exp((vv-theta_m)/sig_m))));
h_nulls = 1 ./ (1+exp((vv-theta_h)./sig_h));

plot(vv,v_nulls,'r');
plot(vv,h_nulls,'b');

%stationary quad

stat_min_v = -59.05;
stat_max_v = -58.95;
stat_min_w = 0.56;
stat_max_w = 0.54;

plot([stat_a(1) stat_b(1)],[stat_a(2) stat_b(2)],'g');
plot([stat_b(1) stat_c(1)],[stat_b(2) stat_c(2)],'g');
plot([stat_c(1) stat_d(1)],[stat_c(2) stat_d(2)],'g');
plot([stat_d(1) stat_a(1)],[stat_d(2) stat_a(2)],'g');

fprintf(revId, '</Mapping>\n');
fclose(revId);
fprintf(jmpId, '</Transitions>\n');
fprintf(jmpId, '</Jump>\n');
fclose(jmpId);

fprintf(jmpInhibId, '</Transitions>\n');
fprintf(jmpInhibId, '</Jump>\n');
fclose(jmpInhibId);

fprintf(jmpBigId, '</Transitions>\n');
fprintf(jmpBigId, '</Jump>\n');
fclose(jmpBigId);

outId = fopen('rinzel.stat', 'w');
fprintf(outId, '<Stationary>\n');
fprintf(outId, '<Quadrilateral><vline>%f %f %f %f</vline><wline>%f %f %f %f</wline></Quadrilateral>\n', stat_a(1), stat_b(1), stat_c(1), stat_d(1), stat_a(2), stat_b(2), stat_c(2), stat_d(2));
fprintf(outId, '</Stationary>\n');
fclose(outId);
