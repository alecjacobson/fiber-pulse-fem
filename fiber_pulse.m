% FIBER_PULSE  Visible-light pulse in a curved dielectric channel
% ("2D slab-waveguide analogue", out-of-plane E_z / TE polarisation).
%
% Same domain as test_single.m (unit square with a bumped channel), but in SI
% units and with a first-order Maxwell system instead of a scalar wave
% equation:
%
%   eps dE_z/dt = d_x H_y - d_y H_x - eps*alpha*E_z - J_z
%   mu0 dH_x/dt = -d_y E_z - mu0*alpha*H_x
%   mu0 dH_y/dt =  d_x E_z - mu0*alpha*H_y - M_y
%
% Discretisation
%   E_z   P1 (piecewise linear) on mesh vertices, lumped mass
%   H     piecewise constant on triangles (H = rotated gradient of E_z)
%   S     S = E x H = (-E_z H_y, E_z H_x) on triangles, with E_z averaged over
%         the triangle and over the leapfrog half step so E and H are co-timed
%   time  explicit leapfrog (E at n dt, H at (n+1/2) dt)
%
% Loading: a Gaussian-envelope pulse at lambda0 matched to one guided TE mode
% of the straight slab, injected one-way by an electric + magnetic current
% sheet. Impedance-matched absorbing layers (equal electric and magnetic loss
% rates) on all domain edges.
%
% Outputs: fiber_pulse.mp4, fiber_pulse_poynting.png,
%   fiber_pulse.mat         mesh (V in metres, F), material, parameters, <S>
%   fiber_pulse_fields/frame_%04d.mat  one file per animation frame (single):
%       fr_E (nV x 1) at fr_t;  fr_Hx,fr_Hy,fr_Sx,fr_Sy (nF x 1) at fr_t-dt/2;  fr_t [s]
%       (one file per frame keeps write cost constant as frames accumulate)
%
% Run headless:  matlab -nodisplay -batch fiber_pulse
% Quick preview: matlab -nodisplay -batch "quick=true; fiber_pulse"

if ~exist('quick','var'), quick = false; end

%% Physical parameters (SI)
c0   = 299792458;
mu0  = 4e-7*pi;
Z0   = mu0*c0;

Lx          = 25e-6;      % domain edge length [m]
core_width  = 0.5e-6;     % channel (core) thickness [m]
lambda0     = 532e-9;     % vacuum wavelength (green) [m]
n_core      = 2.04;       % silicon nitride at 532 nm
n_clad      = 1.46;       % fused silica at 532 nm
eps_core    = n_core^2;   % relative permittivities
eps_clad    = n_clad^2;
mode_order  = 0;          % 0 = fundamental TE mode of the slab
E0          = 1e6;        % peak E_z of the launched mode [V/m]
pulse_sigma = 8e-15;      % Gaussian envelope std dev [s] (FWHM ~ 19 fs)
t_end       = 280e-15;    % simulated time [s]
frame_dt    = 1.5e-15;    % time between animation frames [s]

% Mesh sizing [m]
h_core = 0.025e-6;        % ~ lambda_core/10
h_clad = 0.060e-6;
if quick
  h_core = 0.05e-6; h_clad = 0.15e-6; frame_dt = 4e-15; t_end = 120e-15;
end

% Source plane and absorbing layers [m]
x_src   = 1.5e-6;
sigma_g = 0.1e-6;         % thickness of the (smoothed) source sheet
w_abs   = struct('left',0.8e-6,'right',3e-6,'bottom',3e-6,'top',3e-6);
refl    = 1e-4;           % target round-trip amplitude reflection of absorbers

omega = 2*pi*c0/lambda0;
k0    = omega/c0;

%% Guided mode of the straight slab
d = core_width;
md = slab_mode(k0,d,n_core,n_clad,mode_order);
dk = 1e-4*k0;
mp = slab_mode(k0+dk,d,n_core,n_clad,mode_order);
mm = slab_mode(k0-dk,d,n_core,n_clad,mode_order);
v_group = c0*(2*dk)/(mp.beta-mm.beta);
fprintf('slab V = %.2f (%d guided TE modes), launching TE%d: n_eff = %.4f, n_g = %.4f\n', ...
  md.V, ceil(2*md.V/pi), mode_order, md.neff, c0/v_group);

%% Geometry (identical construction to test_single.m, in unit coordinates)
P = [0 0;1 0;2 1;3 1;4 1;5 0;6 0]/6+[0 0.4166666666666];
C = [ 1 2 3 4;4  5 6 7];
th = 0.5*core_width/Lx;
tol = 1e-2;
sptol = 2e-5;

[oP,oC] = spline_offset(P,C,th,tol);
[uP,uC] = spline_offset(P,C,-th,tol);
oC = flipud(fliplr(oC));

[oV,oE] = spline_to_poly(oP,oC,sptol);
[oI,~,oK] = edges_to_path(oE);
if oK(1) == 2, oI = flipud(oI); end
oV = oV(oI,:);
[uV,uE] = spline_to_poly(uP,uC,sptol);
[uI,~,uK] = edges_to_path(uE);
if uK(1) == 2, uI = flipud(uI); end
uV = uV(uI,:);
aV = [oV;uV];
aV(abs(aV(:,1) - 0)<1e-6,1) = 0;
aV(abs(aV(:,1) - 1)<1e-6,1) = 1;

% Subdivide the core outline to the target core resolution
[RV,RE] = subdivide_loop(aV,h_core/Lx);

% Hex lattice of interior points so the core is resolved across its thickness
hu = h_core/Lx;
[X,Y] = meshgrid(0:hu:1, 0.38:hu*sqrt(3)/2:0.62);
X(2:2:end,:) = X(2:2:end,:) + hu/2;
Q = [X(:) Y(:)];
Q = Q(abs(winding_number(RV,RE,Q))>0.5,:);
Q = Q(point_mesh_squared_distance(Q,RV,RE) > (0.6*hu)^2,:);

max_area = 0.5*(h_clad/Lx)^2;
[V,F] = triangulate([RV;Q;0 0;1 0;1 1;0 1],RE, ...
  'Flags',sprintf('-cq30a%0.17f',max_area));
BC = barycenter(V,F);
in_core = abs(winding_number(RV,RE,BC))>0.5;
fprintf('mesh: %d vertices, %d triangles, %d core triangles\n', ...
  size(V,1),size(F,1),nnz(in_core));

% To metres
V  = V*Lx;
BC = BC*Lx;
nV = size(V,1);
nF = size(F,1);
a  = doublearea(V,F)/2;                       % triangle areas [m^2]
epsr = eps_clad + in_core*(eps_core-eps_clad);

% lengths in the mesh
el = edge_lengths(V,F);
fprintf('edge lengths: min %.1f nm, max %.1f nm\n',1e9*min(el(:)),1e9*max(el(:)));

%% Operators
G  = grad(V,F);
Gx = G(1:nF,:);
Gy = G(nF+1:end,:);
Wx = Gx'*spdiags(a,0,nF,nF);
Wy = Gy'*spdiags(a,0,nF,nF);
% Lumped masses: eps-weighted (E update) and plain volume (sources)
vol  = accumarray(F(:),repmat(a/3,3,1),[nV 1]);
me   = (1/(mu0*c0^2))*accumarray(F(:),repmat(a.*epsr/3,3,1),[nV 1]);
Avg  = sparse(repelem((1:nF)',3),reshape(F',[],1),1/3,nF,nV);

% Leapfrog stability: dt <= 2/sqrt(lambda_max(me^-1 L)), L = (Gx'aGx+Gy'aGy)/mu0
L = (Wx*Gx + Wy*Gy)/mu0;
lam = eigs(L,spdiags(me,0,nV,nV),1,'largestabs','Tolerance',1e-4);
dt = 0.95*2/sqrt(lam);
nsteps = ceil(t_end/dt);
fprintf('dt = %.3g fs, %d steps\n',dt*1e15,nsteps);

%% Absorbing layers (equal electric/magnetic loss rate => impedance matched)
v_abs = c0/n_core;
rate = @(x,y) sponge_rate(x,y,Lx,w_abs,3*v_abs*log(1/refl)/2);
alphaE = rate(V(:,1),V(:,2));
alphaH = rate(BC(:,1),BC(:,2));
cE = (1-alphaE*dt/2)./(1+alphaE*dt/2);
dE = dt./(me.*(1+alphaE*dt/2));
cH = (1-alphaH*dt/2)./(1+alphaH*dt/2);
dH = (dt/mu0)./(1+alphaH*dt/2);

%% One-way mode-matched sheet source
% Centre line and tangent of the channel at x_src (analytic Bezier, first piece)
tt = linspace(0,1,20001)';
B  = (1-tt).^3.*P(1,:)+3*(1-tt).^2.*tt.*P(2,:)+3*(1-tt).*tt.^2.*P(3,:)+tt.^3.*P(4,:);
dB = 3*(1-tt).^2.*(P(2,:)-P(1,:))+6*(1-tt).*tt.*(P(3,:)-P(2,:))+3*tt.^2.*(P(4,:)-P(3,:));
yc  = interp1(B(:,1),B(:,2),x_src/Lx)*Lx;
tan_dir = [interp1(B(:,1),dB(:,1),x_src/Lx) interp1(B(:,1),dB(:,2),x_src/Lx)];
tan_dir = tan_dir/norm(tan_dir);
nrm_dir = [-tan_dir(2) tan_dir(1)];
fprintf('source at x = %.1f um, guide tilt %.1f deg\n',x_src*1e6,atan2d(tan_dir(2),tan_dir(1)));
q_max = d/2 + log(1e4)/md.gamma;

gauss = @(x) exp(-(x-x_src).^2/(2*sigma_g^2))/(sigma_g*sqrt(2*pi));
src_nodes = find(abs(V(:,1)-x_src)<4*sigma_g & abs((V-[x_src yc])*nrm_dir')<q_max);
src_tris  = find(abs(BC(:,1)-x_src)<4*sigma_g & abs((BC-[x_src yc])*nrm_dir')<q_max);
sn = make_src(V(src_nodes,:),[x_src yc],tan_dir,nrm_dir,md,gauss);
st = make_src(BC(src_tris,:),[x_src yc],tan_dir,nrm_dir,md,gauss);
t0 = 4.5*pulse_sigma;
pulse = struct('t0',t0,'sigma',pulse_sigma,'omega',omega,'vg',v_group,'E0',E0,'mu0',mu0, ...
  'beta',md.beta,'cos',tan_dir(1),'sin',tan_dir(2));

%% Display setup
y_win = [0.39 0.61]*Lx;
selF = find(BC(:,2)>=y_win(1) & BC(:,2)<=y_win(2));
[SV,~,SJ,SF] = remove_unreferenced(V,F(selF,:));
SVu = SV*1e6;
% sparse arrow locations (one triangle per 0.5 um cell)
[~,ia] = unique(round(BC(selF,:)/0.5e-6),'rows');
arrow_tri = selF(ia);
arrow_xy = BC(arrow_tri,:)*1e6;
S0 = E0^2*md.neff/Z0;           % peak |S| of the launched mode [W/m^2]
H0 = E0*md.neff/Z0;             % peak |H| [A/m]
arrow_len = 0.45;               % um for peak magnitude

fig = figure('Visible','off','Position',[0 0 1500 900],'Color','w');
tl = tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
cmap_E = flipud(cbrewer('RdBu',256));
loop_u = [RV;RV(1,:)]*Lx*1e6;
ax = gobjects(3,1);
for k = 1:3
  ax(k) = nexttile(tl); hold(ax(k),'on');
end
pE = patch(ax(1),'Faces',SF,'Vertices',SVu,'FaceVertexCData',zeros(size(SV,1),1), ...
  'FaceColor','interp','EdgeColor','none');
pH = patch(ax(2),'Faces',SF,'Vertices',SVu,'FaceVertexCData',zeros(numel(selF),1), ...
  'FaceColor','flat','EdgeColor','none');
pS = patch(ax(3),'Faces',SF,'Vertices',SVu,'FaceVertexCData',zeros(numel(selF),1), ...
  'FaceColor','flat','EdgeColor','none');
qH = quiver(ax(2),arrow_xy(:,1),arrow_xy(:,2),0*arrow_xy(:,1),0*arrow_xy(:,1),0,'k','MaxHeadSize',0.6);
qS = quiver(ax(3),arrow_xy(:,1),arrow_xy(:,2),0*arrow_xy(:,1),0*arrow_xy(:,1),0,'k','MaxHeadSize',0.6);
titles = {'E_z  [MV/m]','|H|  [kA/m] (arrows: H direction & magnitude)','|S| = |E \times H|  [GW/m^2] (arrows: Poynting vector)'};
for k = 1:3
  plot(ax(k),loop_u(:,1),loop_u(:,2),'-','Color',[0 0.4 0],'LineWidth',0.5);
  axis(ax(k),'equal'); xlim(ax(k),[0 Lx*1e6]); ylim(ax(k),y_win*1e6);
  ylabel(ax(k),'y [\mum]'); colorbar(ax(k));
  title(ax(k),titles{k});
end
xlabel(ax(3),'x [\mum]');
colormap(ax(1),cmap_E);  clim(ax(1),[-1 1]*1.1*E0/1e6);
colormap(ax(2),flipud(gray(256))); clim(ax(2),[0 H0/1e3]);
colormap(ax(3),hot(256));  clim(ax(3),[0 S0/1e9]);
sgt = sgtitle(tl,'');

frame_every = max(1,round(frame_dt/dt));
frame_dir = 'fiber_pulse_fields';
if ~exist(frame_dir,'dir'), mkdir(frame_dir); end
kf = 0;
frame_png = [tempname '.png'];
vw = VideoWriter('fiber_pulse.mp4','MPEG-4'); vw.FrameRate = 24; vw.Quality = 95; open(vw);

%% Time stepping
E = zeros(nV,1); Hx = zeros(nF,1); Hy = zeros(nF,1);
Sx_acc = zeros(nF,1); Sy_acc = zeros(nF,1);
My = zeros(nF,1);
tic;
for n = 0:nsteps-1
  t_n = n*dt;
  % H^{n+1/2} from E^n (magnetic sheet current M_y at t_n)
  [Ez_des,~] = sheet_fields(t_n,st,pulse);
  My(src_tris) = Ez_des.*st.g;
  Hx = cH.*Hx - dH.*(Gy*E);
  Hy = cH.*Hy + dH.*(Gx*E - My);
  % E^{n+1} from H^{n+1/2} (electric sheet current J_z at t_{n+1/2})
  [~,Hy_des] = sheet_fields(t_n+dt/2,sn,pulse);
  rhs = Wy*Hx - Wx*Hy;
  rhs(src_nodes) = rhs(src_nodes) - vol(src_nodes).*Hy_des.*sn.g;
  E_old = E;
  E = cE.*E + dE.*rhs;
  % Poynting vector at t_{n+1/2}
  Em = Avg*(0.5*(E+E_old));
  Sx = -Em.*Hy;  Sy = Em.*Hx;
  Sx_acc = Sx_acc + dt*Sx;  Sy_acc = Sy_acc + dt*Sy;

  if mod(n,frame_every)==0
    pE.FaceVertexCData = E(SJ)/1e6;
    Hm = hypot(Hx(selF),Hy(selF));
    Sm = hypot(Sx(selF),Sy(selF));
    pH.FaceVertexCData = Hm/1e3;
    pS.FaceVertexCData = Sm/1e9;
    sc = arrow_len/H0;
    set(qH,'UData',sc*Hx(arrow_tri),'VData',sc*Hy(arrow_tri));
    sc = arrow_len/S0;
    set(qS,'UData',sc*Sx(arrow_tri),'VData',sc*Sy(arrow_tri));
    sgt.String = sprintf('\\lambda_0 = %g nm, TE_%d pulse, t = %6.1f fs',lambda0*1e9,mode_order,(t_n+dt)*1e15);
    kf = kf + 1;
    fr_E = single(E); fr_Hx = single(Hx); fr_Hy = single(Hy);
    fr_Sx = single(Sx); fr_Sy = single(Sy); fr_t = t_n+dt;
    save(fullfile(frame_dir,sprintf('frame_%04d.mat',kf)),'-v6','fr_E','fr_Hx','fr_Hy','fr_Sx','fr_Sy','fr_t');
    exportgraphics(fig,frame_png,'Resolution',96);
    writeVideo(vw,imread(frame_png));
  end
  if mod(n,500)==0
    fprintf('step %d/%d  t = %.1f fs  max|E| = %.3g V/m  (%.0f s)\n',n,nsteps,t_n*1e15,max(abs(E)),toc);
    if ~isfinite(max(abs(E))) || max(abs(E)) > 1e3*E0
      error('fiber_pulse:unstable','Simulation blew up (max|E| = %g).',max(abs(E)));
    end
  end
end
close(vw);
T_total = nsteps*dt;
Sx_avg = Sx_acc/T_total;  Sy_avg = Sy_acc/T_total;

%% Energy-flux sanity check: fluence through vertical strips
for xc = [0.9 3 10 18 21]*1e-6
  strip = abs(BC(:,1)-xc)<0.15e-6;
  fl = sum(Sx_acc(strip).*a(strip))/(0.3e-6);
  fprintf('net x-fluence through x = %4.1f um: %9.3e J/m\n',xc*1e6,fl);
end
fprintf('max|E| at end = %.3g V/m (should be ~0 after absorption)\n',max(abs(E)));

%% Time-averaged Poynting vector figure
Sm_avg = hypot(Sx_avg,Sy_avg);
fig2 = figure('Visible','off','Position',[0 0 1500 1000],'Color','w');
tl2 = tiledlayout(fig2,2,1,'TileSpacing','compact','Padding','compact');
zoom_box = struct('x',[1 9]*1e-6,'y',[0.405 0.595]*Lx);   % first bend
views = {[0 Lx],y_win; zoom_box.x,zoom_box.y};
cell_sz = [0.5e-6 0.15e-6];
for k = 1:2
  axk = nexttile(tl2); hold(axk,'on');
  patch(axk,'Faces',SF,'Vertices',SVu,'FaceVertexCData',log10(max(Sm_avg(selF),1e-4*S0)/S0), ...
    'FaceColor','flat','EdgeColor','none');
  colormap(axk,turbo(256)); clim(axk,[-4 0]);
  cb = colorbar(axk); cb.Label.String = 'log_{10}( |<S>| / S_0 )';
  % arrows: one triangle per cell, only where the flow is significant
  win = BC(:,1)>=views{k,1}(1) & BC(:,1)<=views{k,1}(2) & BC(:,2)>=views{k,2}(1) & BC(:,2)<=views{k,2}(2);
  idx = find(win & Sm_avg>1e-2*S0);
  [~,ib] = unique(round(BC(idx,:)/cell_sz(k)),'rows');
  idx = idx(ib);
  ul = 0.8*cell_sz(k)*1e6;
  quiver(axk,BC(idx,1)*1e6,BC(idx,2)*1e6,ul*Sx_avg(idx)./Sm_avg(idx),ul*Sy_avg(idx)./Sm_avg(idx),0,'k','MaxHeadSize',0.5,'LineWidth',0.5);
  plot(axk,loop_u(:,1),loop_u(:,2),'w-','LineWidth',0.5);
  axis(axk,'equal'); xlim(axk,views{k,1}*1e6); ylim(axk,views{k,2}*1e6);
  xlabel(axk,'x [\mum]'); ylabel(axk,'y [\mum]');
  if k == 1
    title(axk,sprintf('Time-averaged Poynting vector <S> = (1/T)\\int S dt, T = %.0f fs  (S_0 = %.2g W/m^2; arrows: direction)',T_total*1e15,S0));
  else
    title(axk,'Zoom: launch region and first bend');
  end
end
exportgraphics(fig2,'fiber_pulse_poynting.png','Resolution',150);

save('fiber_pulse.mat','V','F','in_core','epsr','dt','nsteps','frame_every','md','pulse','Z0','Sx_avg','Sy_avg','T_total','lambda0','n_core','n_clad','Lx','core_width','-v7.3');
% v6 copy of the summary data for python (scipy.io.loadmat); see render_fiber_pulse.py
neff = md.neff; beta = md.beta; kappa = md.kappa; gamma = md.gamma; order = md.order;
in_core = double(in_core);
save('fiber_pulse_mesh_v6.mat','-v6','V','F','in_core','Sx_avg','Sy_avg','T_total','Lx','E0','neff', ...
  'beta','kappa','gamma','core_width','Z0','lambda0','order');
fprintf('wrote fiber_pulse.mp4, fiber_pulse_poynting.png, fiber_pulse.mat, fiber_pulse_mesh_v6.mat\n');

%% Local functions
function m = slab_mode(k0,d,n1,n2,order)
  % Guided TE mode of a symmetric slab of thickness d (E_z polarisation).
  V = k0*d/2*sqrt(n1^2-n2^2);
  if order >= 2*V/pi
    error('fiber_pulse:mode','Slab with V = %.2f supports only %d TE modes; requested order %d.',V,ceil(2*V/pi),order);
  end
  lo = order*pi/2 + 1e-9;
  hi = min((order+1)*pi/2,V) - 1e-9;
  if mod(order,2) == 0
    f = @(u) u.*tan(u) - sqrt(max(V^2-u.^2,0));
  else
    f = @(u) -u./tan(u) - sqrt(max(V^2-u.^2,0));
  end
  u = fzero(f,[lo hi]);
  m.V = V; m.order = order; m.d = d;
  m.kappa = 2*u/d;
  m.gamma = 2*sqrt(V^2-u^2)/d;
  m.beta = sqrt((k0*n1)^2 - m.kappa^2);
  m.neff = m.beta/k0;
end

function [psi,dpsi] = mode_profile(q,m)
  % Transverse profile of E_z (peak ~1) and its q-derivative.
  in = abs(q) <= m.d/2;
  psi = zeros(size(q)); dpsi = psi;
  if mod(m.order,2) == 0
    edge = cos(m.kappa*m.d/2);
    psi(in)  = cos(m.kappa*q(in));
    dpsi(in) = -m.kappa*sin(m.kappa*q(in));
    out_sign = ones(size(q));
  else
    edge = sin(m.kappa*m.d/2);
    psi(in)  = sin(m.kappa*q(in));
    dpsi(in) = m.kappa*cos(m.kappa*q(in));
    out_sign = sign(q);
  end
  dec = edge*exp(-m.gamma*(abs(q)-m.d/2));
  psi(~in)  = out_sign(~in).*dec(~in);
  dpsi(~in) = -m.gamma*dec(~in);
end

function s = make_src(Pts,c,tan_dir,nrm_dir,md,gauss)
  rel = Pts - c;
  s.s = rel*tan_dir';
  q = rel*nrm_dir';
  [s.psi,s.dpsi] = mode_profile(q,md);
  s.g = gauss(Pts(:,1));
end

function [Ez,Hy] = sheet_fields(t,s,p)
  % Desired forward-propagating mode fields on the source sheet at time t.
  tau = t - p.t0 - s.s/p.vg;
  env = exp(-tau.^2/(2*p.sigma^2));
  ph  = p.omega*(t-p.t0) - p.beta*s.s;
  Ez  = p.E0*s.psi.*env.*cos(ph);
  Hp  = -(p.beta/(p.omega*p.mu0))*Ez;                    % transverse H
  Hd  = -p.E0*s.dpsi.*env.*sin(ph)/(p.mu0*p.omega);      % longitudinal H
  Hy  = Hp*p.cos + Hd*p.sin;
end

function al = sponge_rate(x,y,Lx,w,amax_scale)
  % Quadratic loss ramp inside a layer along each domain edge [1/s].
  ramp = @(depth,wd) amax_scale/wd*(max(depth,0)/wd).^2;
  al = ramp(w.left-x,w.left) + ramp(x-(Lx-w.right),w.right) ...
     + ramp(w.bottom-y,w.bottom) + ramp(y-(Lx-w.top),w.top);
end

function [RV,RE] = subdivide_loop(V,h)
  % Subdivide each edge of a closed polygon to length <= h, keeping vertices.
  n = size(V,1);
  parts = cell(n,1);
  for i = 1:n
    a = V(i,:); b = V(mod(i,n)+1,:);
    k = max(1,ceil(norm(b-a)/h));
    parts{i} = a + ((0:k-1)'/k).*(b-a);
  end
  RV = vertcat(parts{:});
  RE = [1:size(RV,1); [2:size(RV,1) 1]]';
end
